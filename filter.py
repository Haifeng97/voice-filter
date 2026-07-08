import argparse
import asyncio
import base64
import json
import random
import re
import subprocess
import tempfile
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tqdm import tqdm


DEFAULT_ROOT = "/nfs/haifengjia/DuplexConv_stage1_keep"
DEFAULT_OUTPUT = "/data/haifengjia/filter/duplexconv_speaker_profile.jsonl"


SYSTEM_PROMPT = (
    "你是一名语音说话人描述助手。"
    "你的任务是根据音频生成自然中文描述，"
    "重点描述听感性别、年龄感知、声音特点、说话方式和当前交流印象。"
    "必须严格输出 JSON。"
    "不要输出 Markdown。"
    "不要输出 JSON 以外的任何内容。"
)


USER_PROMPT = """
请根据音频中主要说话人的声音和说话方式，输出 JSON。

只输出 JSON，不要输出 Markdown，不要输出解释，不要输出 JSON 以外的任何内容。

输出结构如下：

{
  "transcript": "音频转写文本，听不清则为空字符串",
  "speaker_description": "一句自然中文描述，必须包含：听感性别、年龄感知、声音特点、说话方式或性格印象",
  "perceived_gender": "male | female | child | uncertain",
  "age_estimate_text": "例如：20岁左右、30岁左右、儿童、青少年、中年、年龄不确定但听感偏年轻",
  "summary": "一句自然中文总结"
}

要求：
1. perceived_gender 必须给出听感判断。若非常不确定，使用 uncertain，但 summary 里要写清楚“性别不太确定，听感偏……”。
2. age_estimate_text 必须写成自然中文，不要只写枚举。可以写“20岁左右”“30岁左右”“年龄不确定但听感偏年轻”“像儿童或青少年”。
3. speaker_description 和 summary 都要自然、好读，不要像表格字段。
4. summary 推荐格式：
   “听感上像一位女性，年龄可能在20岁左右；声音偏高，语速较慢，语气可爱、撒娇，像是在和亲近的人开玩笑。”
5. 年龄和性别都只是根据声音得到的感知结果，不代表真实身份。
6. 性格只描述这段音频里呈现出的交流印象，不代表长期人格。
7. 不要推断职业、学历、地域、民族、健康状况、真实身份。
8. summary 不超过 120 个中文字。
""".strip()


def run_cmd(cmd: list[str], timeout: int | float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def get_wav_info(path: Path) -> dict[str, Any]:
    """
    优先用 wave 读 WAV 头，快。
    失败时用 ffprobe 兜底。
    """
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            frames = wf.getnframes()
            duration = frames / sample_rate if sample_rate > 0 else None
            return {
                "channels": channels,
                "sample_rate": sample_rate,
                "duration": duration,
            }
    except Exception:
        pass

    proc = run_cmd([
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=channels,sample_rate,duration",
        "-of", "json",
        str(path),
    ])

    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()}")

    data = json.loads(proc.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe found no audio stream")

    s = streams[0]
    duration = s.get("duration")
    try:
        duration = float(duration) if duration is not None else None
    except Exception:
        duration = None

    return {
        "channels": int(s.get("channels", 1)),
        "sample_rate": int(s.get("sample_rate", 0)),
        "duration": duration,
    }


def parse_api_bases(values: list[str]) -> list[str]:
    out = []
    for v in values:
        for part in v.split(","):
            part = part.strip()
            if not part:
                continue
            part = part.rstrip("/")
            if not part.endswith("/v1"):
                part = part + "/v1"
            out.append(part)

    if not out:
        out = ["http://127.0.0.1:18000/v1"]

    return out


def load_done_keys(output_path: Path, retry_failed: bool) -> set[str]:
    """
    resume 用。
    默认：已有记录都跳过。
    --retry-failed：只跳过 parse_ok=true 的记录，失败记录会重跑。
    """
    done = set()

    if not output_path.is_file():
        return done

    with output_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            wav_path = obj.get("wav_path")
            channel = obj.get("channel")
            if wav_path is None or channel is None:
                continue

            if retry_failed and not obj.get("parse_ok", False):
                continue

            done.add(f"{wav_path}|{channel}")

    return done


def collect_wav_paths(root: Path, input_list: Path | None) -> list[Path]:
    if input_list is not None:
        paths = []
        with input_list.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    paths.append(Path(line))
        return paths

    return list(root.rglob("*.wav"))


def parse_channels_arg(channels_arg: str, num_channels: int) -> list[int]:
    if channels_arg == "all":
        return list(range(num_channels))

    channels = []
    for x in channels_arg.split(","):
        x = x.strip()
        if not x:
            continue
        ch = int(x)
        if 0 <= ch < num_channels:
            channels.append(ch)

    return channels


def build_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = Path(args.root)
    input_list = Path(args.input_list) if args.input_list else None
    output_path = Path(args.output)

    done = load_done_keys(output_path, retry_failed=args.retry_failed)

    wav_paths = collect_wav_paths(root, input_list)

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(wav_paths)
    else:
        wav_paths = sorted(wav_paths)

    tasks = []

    for wav_path in wav_paths:
        try:
            info = get_wav_info(wav_path)
            num_channels = int(info["channels"])
        except Exception as e:
            err_key = f"{wav_path}|probe_failed"
            if err_key not in done:
                tasks.append({
                    "wav_path": str(wav_path),
                    "channel": None,
                    "source_channels": None,
                    "source_sample_rate": None,
                    "source_duration": None,
                    "probe_error": repr(e),
                    "probe_failed": True,
                })
            continue

        channels = parse_channels_arg(args.channels, num_channels)

        for ch in channels:
            key = f"{wav_path}|{ch}"
            if key in done:
                continue

            tasks.append({
                "wav_path": str(wav_path),
                "channel": ch,
                "source_channels": num_channels,
                "source_sample_rate": info.get("sample_rate"),
                "source_duration": info.get("duration"),
                "probe_failed": False,
            })

            if args.limit is not None and len(tasks) >= args.limit:
                return tasks

    return tasks


def extract_channel_to_temp(
    wav_path: str,
    channel: int,
    segment_start: float,
    segment_duration: float,
) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        prefix="stepaudio_profile_",
        suffix=".wav",
        delete=False,
    )
    tmp_path = Path(tmp.name)
    tmp.close()

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
    ]

    if segment_start > 0:
        cmd += ["-ss", str(segment_start)]

    cmd += [
        "-i", wav_path,
        "-map", "0:a:0",
        "-af", f"pan=mono|c0=c{channel}",
        "-ar", "16000",
        "-ac", "1",
        "-t", str(segment_duration),
        str(tmp_path),
    ]

    proc = run_cmd(cmd)

    if proc.returncode != 0:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")

    return tmp_path


def extract_json_obj(text: str) -> dict[str, Any]:
    cleaned = text.strip()

    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```.*$", "", cleaned, flags=re.S)

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start < 0 or end <= start:
        raise ValueError("没有找到 JSON 对象")

    cleaned = cleaned[start:end + 1]
    return json.loads(cleaned)


def normalize_result(obj: dict[str, Any]) -> dict[str, Any]:
    allowed_gender = {"male", "female", "child", "uncertain"}

    gender = obj.get("perceived_gender", "uncertain")
    if gender not in allowed_gender:
        gender = "uncertain"

    transcript = str(obj.get("transcript", "") or "").strip()
    speaker_description = str(obj.get("speaker_description", "") or "").strip()
    age_estimate_text = str(obj.get("age_estimate_text", "") or "").strip()
    summary = str(obj.get("summary", "") or "").strip()

    return {
        "transcript": transcript,
        "speaker_description": speaker_description[:200],
        "perceived_gender": gender,
        "age_estimate_text": age_estimate_text[:80],
        "summary": summary[:200],
    }


async def call_stepaudio(
    client: AsyncOpenAI,
    model: str,
    audio_path: Path,
    max_tokens: int,
    timeout: float,
) -> str:
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("utf-8")

    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": f"data:audio/wav;base64,{audio_b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": USER_PROMPT,
                        },
                    ],
                },
            ],
            extra_body={
                "sampling_params_list": [
                    {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "top_k": -1,
                        "max_tokens": max_tokens,
                        "seed": 42,
                        "detokenize": True,
                        "repetition_penalty": 1.05,
                        "stop_token_ids": [151645],
                    }
                ]
            },
        ),
        timeout=timeout,
    )

    return response.choices[0].message.content or ""


async def process_one(
    job: dict[str, Any],
    idx: int,
    args: argparse.Namespace,
    api_bases: list[str],
    clients: dict[str, AsyncOpenAI],
    endpoint_sems: dict[str, asyncio.Semaphore],
) -> dict[str, Any]:
    start_time = time.time()

    base_record = {
        "task_index": idx,
        "wav_path": job.get("wav_path"),
        "channel": job.get("channel"),
        "source_channels": job.get("source_channels"),
        "source_sample_rate": job.get("source_sample_rate"),
        "source_duration": job.get("source_duration"),
        "segment_start": args.segment_start,
        "segment_duration": args.segment_duration,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    if job.get("probe_failed"):
        return {
            **base_record,
            "parse_ok": False,
            "error": job.get("probe_error"),
            "raw_model_output": "",
            "elapsed_sec": round(time.time() - start_time, 3),
        }

    api_base = api_bases[idx % len(api_bases)]
    base_record["api_base"] = api_base

    last_error = None
    raw_text = ""

    for attempt in range(args.retries + 1):
        tmp_audio = None

        try:
            tmp_audio = extract_channel_to_temp(
                wav_path=job["wav_path"],
                channel=int(job["channel"]),
                segment_start=args.segment_start,
                segment_duration=args.segment_duration,
            )

            async with endpoint_sems[api_base]:
                raw_text = await call_stepaudio(
                    client=clients[api_base],
                    model=args.model,
                    audio_path=tmp_audio,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )

            parsed = extract_json_obj(raw_text)
            normalized = normalize_result(parsed)

            return {
                **base_record,
                **normalized,
                "parse_ok": True,
                "error": "",
                "raw_model_output": raw_text,
                "elapsed_sec": round(time.time() - start_time, 3),
            }

        except Exception as e:
            last_error = repr(e)

            if attempt < args.retries:
                await asyncio.sleep(min(2 ** attempt, 8))

        finally:
            if tmp_audio is not None:
                try:
                    tmp_audio.unlink(missing_ok=True)
                except Exception:
                    pass

    return {
        **base_record,
        "transcript": "",
        "speaker_description": "",
        "perceived_gender": "uncertain",
        "age_estimate_text": "",
        "summary": "",
        "parse_ok": False,
        "error": last_error,
        "raw_model_output": raw_text,
        "elapsed_sec": round(time.time() - start_time, 3),
    }


async def async_main(args: argparse.Namespace) -> None:
    if args.test and args.limit is None:
        args.limit = 20

    api_bases = parse_api_bases(args.api_bases)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args)

    print(f"API endpoints: {api_bases}")
    print(f"Output: {output_path}")
    print(f"Tasks to run: {len(tasks)}")
    print(f"Concurrency: total={args.concurrency}, per_api={args.per_api_concurrency}")
    print(f"Segment: start={args.segment_start}s, duration={args.segment_duration}s")

    if not tasks:
        print("没有需要处理的任务。可能已经全部 resume 跳过了。")
        return

    clients = {
        api: AsyncOpenAI(
            api_key=args.api_key,
            base_url=api,
            timeout=args.timeout,
        )
        for api in api_bases
    }

    endpoint_sems = {
        api: asyncio.Semaphore(args.per_api_concurrency)
        for api in api_bases
    }

    global_sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    ok_count = 0
    fail_count = 0

    async def runner(job: dict[str, Any], idx: int) -> dict[str, Any]:
        async with global_sem:
            return await process_one(
                job=job,
                idx=idx,
                args=args,
                api_bases=api_bases,
                clients=clients,
                endpoint_sems=endpoint_sems,
            )

    pending = [
        asyncio.create_task(runner(job, idx))
        for idx, job in enumerate(tasks)
    ]

    with output_path.open("a", encoding="utf-8", buffering=1) as f:
        with tqdm(total=len(pending), desc="StepAudio profile", dynamic_ncols=True) as pbar:
            for coro in asyncio.as_completed(pending):
                rec = await coro

                async with write_lock:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if rec.get("parse_ok"):
                    ok_count += 1
                else:
                    fail_count += 1

                pbar.set_postfix(ok=ok_count, fail=fail_count)
                pbar.update(1)

    print(f"完成：ok={ok_count}, fail={fail_count}, output={output_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch Step-Audio2 speaker profile over wav channels."
    )

    p.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help=f"wav 根目录，默认 {DEFAULT_ROOT}",
    )
    p.add_argument(
        "--input-list",
        default="",
        help="可选：txt 文件，每行一个 wav 路径。若提供则不扫描 root。",
    )
    p.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"输出 JSONL，默认 {DEFAULT_OUTPUT}",
    )

    p.add_argument(
        "--api-bases",
        nargs="+",
        default=["http://127.0.0.1:18000/v1"],
        help="一个或多个 OpenAI API base。支持空格或逗号分隔。",
    )
    p.add_argument(
        "--model",
        default="step-audio2-mini",
        help="served model name",
    )
    p.add_argument(
        "--api-key",
        default="EMPTY",
    )

    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="总并发数。多卡多端口时可设为 API 数量。",
    )
    p.add_argument(
        "--per-api-concurrency",
        type=int,
        default=1,
        help="每个 API endpoint 的并发。你现在 max_batch_size=1，建议保持 1。",
    )

    p.add_argument(
        "--channels",
        default="all",
        help='处理哪些声道。默认 all。也可以写 "0" 或 "0,1"。',
    )
    p.add_argument(
        "--segment-start",
        type=float,
        default=0.0,
        help="截取片段起点秒数，默认 0。",
    )
    p.add_argument(
        "--segment-duration",
        type=float,
        default=20.0,
        help="截取片段时长，默认 20 秒。不要设 30，容易踩 Step-Audio2 1500 边界。",
    )

    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多处理多少个任务。测试用。",
    )
    p.add_argument(
        "--test",
        action="store_true",
        help="测试模式；若未指定 --limit，则默认跑 20 个任务。",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="随机打乱任务，默认开启。",
    )
    p.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="不随机打乱。",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--retries",
        type=int,
        default=2,
        help="失败重试次数，默认 2。",
    )
    p.add_argument(
        "--retry-failed",
        action="store_true",
        help="resume 时重跑之前 parse_ok=false 的失败任务。",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="单个请求超时时间秒。",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=700,
    )

    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.per_api_concurrency < 1:
        raise ValueError("--per-api-concurrency must be >= 1")
    if args.segment_duration > 25:
        print("警告：segment_duration > 25 可能再次触发 Step-Audio2 音频长度边界问题。")

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
