# Fake ESP32 Jevlike edge router

A tiny Python demo of this repository's text model. The "ESP32" accepts ASCII
text and classifies it into `command`, `weather`, or `complex`.

If it is a command, it acts locally. If it is weather, it prints that the
weather service would be queried. If it is complex, or confidence is low, it
prints that the request would be forwarded to a cloud LLM.

This is intentionally a demo, not a production model.

## Run

From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python examples/esp32_demo/fake_esp32.py --train --device cpu
python examples/esp32_demo/fake_esp32.py --once "turn on the kitchen light"
python examples/esp32_demo/fake_esp32.py
```

`--train` expands the hand-written JSONL files with held-out templates, writes
`runs/esp32_demo/command-router.pt`, and evaluates the generated test split.
The interactive loop loads that checkpoint and routes on the model's top option.
