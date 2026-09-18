#!/usr/bin/env python3
"""Fake ESP32 + Jevlike classifier demo.

Run from the repository root after installing jevlike:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .
    python examples/esp32_demo/fake_esp32.py --train --device cpu
    python examples/esp32_demo/fake_esp32.py --once "turn on the kitchen light"
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import torch

from jevlike.data import ChoiceExample
from jevlike.model import load_checkpoint, select_device
from jevlike.train import move

OPTIONS = ("command", "weather", "complex")
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
RUNS = REPO / "runs" / "esp32_demo"
MODEL = RUNS / "command-router.pt"
HAND_TRAIN = ROOT / "train.jsonl"
HAND_VALIDATION = ROOT / "validation.jsonl"
HAND_TEST = ROOT / "test.jsonl"

# Act locally only when the winning class is this confident.
CONFIDENCE = 0.45


def _row(context: str, label: int) -> dict:
    return {"context": context, "options": list(OPTIONS), "label": label}


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _command_lines(rooms: list[str], devices: list[str]) -> list[str]:
    verbs_on = ("turn on", "switch on")
    verbs_off = ("turn off", "switch off")
    other = (
        "open the {room} window",
        "close the {room} blinds",
        "lock the {room} door",
        "unlock the {room} door",
        "set the {room} thermostat to {temp} degrees",
        "set the {room} volume to {vol}",
        "start the dishwasher",
        "stop the washing machine",
        "dim the {room} lights",
    )
    lines = []
    for room in rooms:
        for device in devices:
            for verb in verbs_on + verbs_off:
                lines.append(f"{verb} the {room} {device}")
                lines.append(f"please {verb} the {room} {device}")
        for template in other:
            lines.append(template.format(room=room, temp=19 + (len(room) % 5), vol=5 + (len(room) % 8)))
    return lines


def _weather_lines(places: list[str], times: list[str]) -> list[str]:
    templates = (
        "what is the weather {time} in {place}",
        "what is the forecast for {place} {time}",
        "will it rain in {place} {time}",
        "is rain likely {time} in {place}",
        "how cold will it be {time} in {place}",
        "how hot will it be {time} in {place}",
        "how warm will {time} be in {place}",
        "do I need an umbrella {time} in {place}",
        "is it sunny in {place} {time}",
        "is snow expected {time} in {place}",
        "check today's weather in {place}",
        "what is the temperature {time} in {place}",
        "what is the wind speed {time} in {place}",
        "humidity in {place} {time}",
        "should I take a coat {time} in {place}",
    )
    lines = []
    for place in places:
        for time in times:
            for template in templates:
                lines.append(template.format(place=place, time=time))
    return lines


def _complex_lines(topics: list[str]) -> list[str]:
    templates = (
        "explain how {topic} works",
        "explain why {topic} matters",
        "why should I use {topic}",
        "help me debug {topic}",
        "compare {topic} with MQTT",
        "write a Python program about {topic}",
        "write a C program that uses {topic}",
        "help me design a system using {topic}",
        "summarise {topic} for a beginner",
        "how does {topic} protect a connection",
        "suggest an architecture that uses {topic}",
        "what is the difference between {topic} and TCP",
    )
    lines = [template.format(topic=topic) for topic in topics for template in templates]
    lines.extend((
        "draft an email to my manager about the project delay",
        "translate this paragraph into French",
        "what is the meaning of life",
        "prove that P is not equal to NP",
        "how does a Kalman filter work",
        "summarise the difference between TCP and UDP",
    ))
    return lines


def build_splits() -> tuple[Path, Path, Path]:
    """Expand the tiny hand-written lists with held-out template fills."""
    train_rooms = ["kitchen", "bedroom", "hallway", "office", "garage"]
    val_rooms = ["living room", "attic"]
    test_rooms = ["bathroom", "porch"]
    devices = ["light", "lamp", "lights", "fan"]

    train_places = ["Prague", "Brno", "Ostrava"]
    val_places = ["London"]
    test_places = ["Vienna", "Berlin"]
    train_times = ["today", "tomorrow", "this evening", "tomorrow morning"]
    val_times = ["tonight", "this weekend"]
    test_times = ["Friday afternoon", "on Saturday"]

    train_topics = ["MQTT", "QUIC", "a hash function", "HTTPS certificates", "public key cryptography", "UART"]
    val_topics = ["AES", "firmware updates"]
    test_topics = ["TLS 1.3", "mutual TLS", "a deadlock in my firmware"]

    def labelled(lines: list[str], label: int) -> list[dict]:
        return [_row(line, label) for line in lines]

    train = (
        labelled(_command_lines(train_rooms, devices), 0)
        + labelled(_weather_lines(train_places, train_times), 1)
        + labelled(_complex_lines(train_topics), 2)
        + _load_jsonl(HAND_TRAIN)
    )
    validation = (
        labelled(_command_lines(val_rooms, devices), 0)
        + labelled(_weather_lines(val_places, val_times), 1)
        + labelled(_complex_lines(val_topics), 2)
        + _load_jsonl(HAND_VALIDATION)
    )
    test = (
        labelled(_command_lines(test_rooms, ["light", "window"]), 0)
        + labelled(_weather_lines(test_places, test_times), 1)
        + labelled(_complex_lines(test_topics), 2)
        + _load_jsonl(HAND_TEST)
    )

    rng = random.Random(7)
    for rows in (train, validation, test):
        rng.shuffle(rows)

    RUNS.mkdir(parents=True, exist_ok=True)
    train_path, val_path, test_path = RUNS / "train.jsonl", RUNS / "validation.jsonl", RUNS / "test.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(val_path, validation)
    _write_jsonl(test_path, test)
    print(json.dumps({
        "train": len(train), "validation": len(validation), "test": len(test),
    }), flush=True)
    return train_path, val_path, test_path


def train(*, epochs: int, device: str, batch_size: int) -> None:
    print("[ESP32] training tiny edge classifier...", flush=True)
    train_path, val_path, test_path = build_splits()
    subprocess.run(
        [
            sys.executable, "-m", "jevlike.train",
            str(train_path),
            "--validation", str(val_path),
            "--output", str(MODEL),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--device", device,
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "jevlike.eval", str(MODEL), str(test_path), "--device", device],
        check=True,
    )


def load_model(device_name: str):
    device = select_device(device_name)
    model, collator, _ = load_checkpoint(MODEL, device)
    return model, collator, device


def predict(model, collator, device, text: str) -> list[dict]:
    batch = move(collator([ChoiceExample(text, OPTIONS, 0)]), device)
    model.eval()
    with torch.no_grad():
        probabilities = model(batch).softmax(-1)[0, :len(OPTIONS)].cpu().tolist()
    return [
        {"option": option, "probability": probability}
        for option, probability in zip(OPTIONS, probabilities)
    ]


def local_action(text: str) -> None:
    print(f"[ESP32] local command: {text!r}")
    lowered = text.lower()
    if "on" in lowered and "off" not in lowered:
        print("[ESP32] GPIO -> LIGHT ON")
    elif "off" in lowered:
        print("[ESP32] GPIO -> LIGHT OFF")
    else:
        print("[ESP32] GPIO/MQTT -> execute command")


def route(text: str, scores: list[dict]) -> None:
    print(json.dumps(scores, indent=2))
    best = max(scores, key=lambda row: row["probability"])
    option = best["option"]
    if best["probability"] < CONFIDENCE:
        print(f"[EDGE] low confidence ({best['probability']:.3f}) -> forward to cloud LLM")
        return
    if option == "weather":
        print("[EDGE] weather -> local weather client / API")
    elif option == "command":
        local_action(text)
    else:
        print("[EDGE] complex -> forward to cloud LLM")


def handle(model, collator, device, text: str) -> None:
    route(text, predict(model, collator, device, text))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fake ESP32 Jevlike edge router")
    parser.add_argument("--train", action="store_true", help="train (or retrain) the classifier")
    parser.add_argument("--once", action="append", default=[], metavar="TEXT",
                        help="classify one utterance and exit; repeatable")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train or not MODEL.exists():
        train(epochs=args.epochs, device=args.device, batch_size=args.batch_size)

    model, collator, device = load_model(args.device)
    if args.once:
        for text in args.once:
            print(f"> {text}")
            handle(model, collator, device, text)
        return

    print("""
Fake ESP32 online.

Type ASCII commands:
  turn on the kitchen light
  what is the weather tomorrow?
  explain how TLS 1.3 works

Ctrl-D/Ctrl-C to exit.
""")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text:
            handle(model, collator, device, text)


if __name__ == "__main__":
    main()
