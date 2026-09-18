#!/usr/bin/env python3
"""Pack a trained TinyScorer checkpoint for the ESP-IDF firmware."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

import torch

from jevlike.model import load_checkpoint, select_device
from jevlike.train import move
from jevlike.data import ChoiceExample

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
MODEL = REPO / "runs" / "esp32_demo" / "command-router.pt"
FIRMWARE = ROOT / "firmware"
WEIGHTS = FIRMWARE / "main" / "weights.bin"
OPTIONS = ("command", "weather", "complex")
DEMOS = (
    "turn on the kitchen light",
    "what is the weather tomorrow?",
    "explain how TLS 1.3 works",
)


def pack(checkpoint: Path, output: Path) -> dict:
    device = select_device("cpu")
    model, collator, config = load_checkpoint(checkpoint, device)
    if config.get("encoder") != "tiny":
        raise SystemExit("firmware export only supports the tiny byte encoder")
    state = model.state_dict()

    def blob(name: str) -> bytes:
        return state[name].detach().contiguous().cpu().to(torch.float32).numpy().tobytes()

    header = struct.pack(
        "<8s6I",
        b"JEVLIKE1",
        int(config["width"]),
        int(config["rank"]),
        int(config["context_tokens"]),
        int(config["option_tokens"]),
        257,
        len(OPTIONS),
    )
    body = b"".join((
        blob("embedding.weight"),
        blob("position.weight"),
        blob("head.context_norm.weight"),
        blob("head.context_norm.bias"),
        blob("head.option_norm.weight"),
        blob("head.option_norm.bias"),
        blob("head.query.weight"),
        blob("head.key.weight"),
        blob("head.value.weight"),
    ))
    names = b"".join(option.encode("ascii") + b"\0" for option in OPTIONS)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(header + body + names)

    refs = []
    model.eval()
    with torch.no_grad():
        for text in DEMOS:
            batch = move(collator([ChoiceExample(text, OPTIONS, 0)]), device)
            probs = model(batch).softmax(-1)[0, :len(OPTIONS)].cpu().tolist()
            refs.append({"context": text, "probability": dict(zip(OPTIONS, probs))})
    return {"checkpoint": str(checkpoint), "weights": str(output), "bytes": output.stat().st_size,
            "config": {k: config[k] for k in ("width", "rank", "context_tokens", "option_tokens")},
            "python": refs}


def host_check(weights: Path, refs: list[dict]) -> None:
    binary = REPO / "runs" / "esp32_demo" / "host_check"
    subprocess.run(
        [
            "gcc", "-O2", "-std=c11",
            "-I", str(FIRMWARE / "main"),
            "-o", str(binary),
            str(FIRMWARE / "host_check.c"),
            str(FIRMWARE / "main" / "jevlike_scorer.c"),
            "-lm",
        ],
        check=True,
    )
    texts = [row["context"] for row in refs]
    result = subprocess.run(
        [str(binary), str(weights), *texts],
        check=True, capture_output=True, text=True,
    )
    print(result.stdout, end="")
    by_text = {row["context"]: row["probability"] for row in refs}
    for line in result.stdout.splitlines():
        text, *pairs = line.split("\t")
        got = dict(item.split("=", 1) for item in pairs)
        for option, expected in by_text[text].items():
            actual = float(got[option])
            if abs(actual - expected) > 2e-5:
                raise SystemExit(
                    f"host/python mismatch for {text!r} {option}: "
                    f"c={actual:.8f} python={expected:.8f}"
                )
    print(json.dumps({"host_check": "ok", "atol": 2e-5}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, default=WEIGHTS)
    parser.add_argument("--skip-host-check", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"missing {args.checkpoint}; train with fake_esp32.py --train first")
    payload = pack(args.checkpoint, args.output)
    print(json.dumps(payload, indent=2))
    if not args.skip_host_check:
        host_check(args.output, payload["python"])


if __name__ == "__main__":
    main()
