#!/usr/bin/env python3
"""Phase 0 benchmark + smoke harness for vib3 OpenAI-compatible API.

This script provides:
- Deterministic smoke checks for correctness/stability regressions
- Baseline performance sampling (latency, TTFT, tok/s, req/s, p95)
- Optional concurrent load for mixed prompt lengths

No third-party dependencies are required.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


SHORT_PROMPTS = [
    "What is 2+2? Answer with one word.",
    "Reply with exactly one word: blue or green?",
    "What comes after A in the alphabet? One word.",
]

MIXED_PROMPTS = [
    "Summarize why batching improves throughput in one sentence.",
    "Explain continuous batching vs static batching in 3 bullets.",
    "Write a compact Python function that computes fibonacci iteratively.",
    "Give 5 short tips to reduce p95 latency in GPU inference services.",
]

LONG_PROMPTS = [
    "You are tuning a Rust/CUDA inference runtime. Explain a practical plan for "
    "prefill/decode scheduling, memory admission control, and CUDA stream overlap. "
    "Provide concrete trade-offs and failure modes."
]

SMOKE_CASES = [
    {
        "name": "math_one_word",
        "prompt": "What is 2+2? Answer with one word.",
        "max_tokens": 12,
        "must_contain_any": ["4", "four"],
    },
    {
        "name": "short_coherent",
        "prompt": "Write one short sentence about the sky.",
        "max_tokens": 24,
        "must_not_be_empty": True,
    },
]


@dataclass
class RequestMetrics:
    ok: bool
    status_code: int
    error: str
    latency_ms: float
    ttft_ms: Optional[float]
    completion_tokens: int
    content: str


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * p
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return values[lo]
    frac = rank - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def choose_prompt(kind: str, idx: int) -> str:
    if kind == "short":
        return SHORT_PROMPTS[idx % len(SHORT_PROMPTS)]
    if kind == "mixed":
        return MIXED_PROMPTS[idx % len(MIXED_PROMPTS)]
    return LONG_PROMPTS[idx % len(LONG_PROMPTS)]


def stream_chat_completion(
    *,
    base_url: str,
    model: str,
    api_key: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
) -> RequestMetrics:
    url = urllib.parse.urljoin(base_url.rstrip("/") + "/", "chat/completions")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    start = time.perf_counter()
    first_token_ts: Optional[float] = None
    content_parts: List[str] = []
    completion_tokens = 0

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = getattr(resp, "status", 200)
            while True:
                raw = resp.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if not chunk or chunk == "[DONE]":
                    continue
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    continue

                usage = obj.get("usage")
                if isinstance(usage, dict):
                    completion_tokens = int(usage.get("completion_tokens") or completion_tokens)

                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = delta.get("content")
                if text:
                    content_parts.append(text)
                    if first_token_ts is None:
                        first_token_ts = time.perf_counter()

            end = time.perf_counter()
            content = "".join(content_parts).strip()
            if completion_tokens == 0 and content:
                completion_tokens = max(1, len(content.split()))
            return RequestMetrics(
                ok=True,
                status_code=int(status),
                error="",
                latency_ms=(end - start) * 1000.0,
                ttft_ms=(None if first_token_ts is None else (first_token_ts - start) * 1000.0),
                completion_tokens=completion_tokens,
                content=content,
            )
    except urllib.error.HTTPError as err:
        end = time.perf_counter()
        body = ""
        try:
            body = err.read().decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        return RequestMetrics(
            ok=False,
            status_code=err.code,
            error=f"HTTPError {err.code}: {body[:400]}",
            latency_ms=(end - start) * 1000.0,
            ttft_ms=None,
            completion_tokens=0,
            content="",
        )
    except Exception as err:
        end = time.perf_counter()
        return RequestMetrics(
            ok=False,
            status_code=0,
            error=str(err),
            latency_ms=(end - start) * 1000.0,
            ttft_ms=None,
            completion_tokens=0,
            content="",
        )


def run_smoke(
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout_s: int,
) -> Dict[str, Any]:
    results = []
    all_passed = True
    for case in SMOKE_CASES:
        metrics = stream_chat_completion(
            base_url=base_url,
            model=model,
            api_key=api_key,
            prompt=case["prompt"],
            max_tokens=int(case["max_tokens"]),
            temperature=0.0,
            timeout_s=timeout_s,
        )
        content_lower = metrics.content.lower()
        passed = metrics.ok
        looks_like_error = (
            content_lower.startswith("[error:")
            or "cuda error" in content_lower
            or "requires gpu" in content_lower
        )
        if looks_like_error:
            passed = False
        if case.get("must_contain_any"):
            needles = [str(v).lower() for v in case["must_contain_any"]]
            passed = passed and any(n in content_lower for n in needles)
        if case.get("must_not_be_empty"):
            passed = passed and bool(metrics.content.strip())
        if not passed:
            all_passed = False

        results.append(
            {
                "name": case["name"],
                "passed": passed,
                "latency_ms": round(metrics.latency_ms, 2),
                "ttft_ms": None if metrics.ttft_ms is None else round(metrics.ttft_ms, 2),
                "content": metrics.content,
                "error": metrics.error,
            }
        )

    return {"all_passed": all_passed, "cases": results}


def run_perf(
    *,
    base_url: str,
    model: str,
    api_key: str,
    scenario: str,
    runs: int,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
) -> Dict[str, Any]:
    started = time.perf_counter()
    metrics_list: List[RequestMetrics] = []
    lock = threading.Lock()

    def one_call(i: int) -> RequestMetrics:
        prompt = choose_prompt(scenario, i)
        return stream_chat_completion(
            base_url=base_url,
            model=model,
            api_key=api_key,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(one_call, i) for i in range(runs)]
        for future in as_completed(futures):
            result = future.result()
            with lock:
                metrics_list.append(result)

    finished = time.perf_counter()
    total_s = max(1e-6, finished - started)

    oks = [m for m in metrics_list if m.ok]
    latencies = sorted(m.latency_ms for m in oks)
    ttfts = sorted(m.ttft_ms for m in oks if m.ttft_ms is not None)
    total_completion_tokens = sum(m.completion_tokens for m in oks)

    req_s = len(metrics_list) / total_s
    tok_s = total_completion_tokens / total_s

    return {
        "scenario": scenario,
        "runs": runs,
        "concurrency": concurrency,
        "success": len(oks),
        "errors": len(metrics_list) - len(oks),
        "req_s": round(req_s, 3),
        "tok_s": round(tok_s, 3),
        "latency_ms_avg": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "latency_ms_p95": round(percentile(latencies, 0.95), 2) if latencies else 0.0,
        "ttft_ms_avg": round(statistics.mean(ttfts), 2) if ttfts else 0.0,
        "ttft_ms_p95": round(percentile(ttfts, 0.95), 2) if ttfts else 0.0,
        "sample_errors": [m.error for m in metrics_list if not m.ok][:5],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 0 vib3 API smoke + benchmark harness")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--model", default="default", help="Served model name")
    parser.add_argument("--api-key", default="not-needed", help="Bearer token")
    parser.add_argument("--runs", type=int, default=24, help="Total requests per scenario")
    parser.add_argument("--concurrency", type=int, default=4, help="Parallel workers")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max completion tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--timeout", type=int, default=120, help="Per-request timeout in seconds")
    parser.add_argument(
        "--scenarios",
        default="short,mixed,long",
        help="Comma-separated list from: short,mixed,long",
    )
    parser.add_argument("--skip-smoke", action="store_true", help="Skip deterministic smoke checks")
    parser.add_argument("--output", default="", help="Optional path for JSON report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenario_names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    allowed = {"short", "mixed", "long"}
    invalid = [s for s in scenario_names if s not in allowed]
    if invalid:
        print(f"Invalid scenarios: {invalid}; allowed={sorted(allowed)}")
        return 2

    report: Dict[str, Any] = {
        "timestamp_unix": int(time.time()),
        "base_url": args.base_url,
        "model": args.model,
        "config": {
            "runs": args.runs,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "timeout": args.timeout,
            "scenarios": scenario_names,
        },
    }

    if not args.skip_smoke:
        smoke = run_smoke(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            timeout_s=args.timeout,
        )
        report["smoke"] = smoke
        print(f"SMOKE all_passed={smoke['all_passed']}")
        for case in smoke["cases"]:
            print(
                f"  - {case['name']}: passed={case['passed']} latency_ms={case['latency_ms']} ttft_ms={case['ttft_ms']}"
            )

    perf_reports = []
    for scenario in scenario_names:
        perf = run_perf(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            scenario=scenario,
            runs=args.runs,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_s=args.timeout,
        )
        perf_reports.append(perf)
        print(
            "PERF"
            f" scenario={perf['scenario']} success={perf['success']}/{perf['runs']}"
            f" req_s={perf['req_s']} tok_s={perf['tok_s']}"
            f" p95_latency_ms={perf['latency_ms_p95']} p95_ttft_ms={perf['ttft_ms_p95']}"
        )
        if perf["sample_errors"]:
            print(f"  sample_errors={perf['sample_errors']}")

    report["performance"] = perf_reports

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Wrote report: {args.output}")
    else:
        print(json.dumps(report, indent=2))

    smoke_ok = report.get("smoke", {}).get("all_passed", True)
    return 0 if smoke_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
