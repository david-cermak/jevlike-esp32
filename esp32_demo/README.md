# ESP32 Jevlike edge router demo

A tiny demo of this jevlike repository's text model. Training stays in Python. The
same scorer then runs either in Python or as an ESP-IDF firmware.

The model classifies ASCII text as `command`, `weather`, or `complex` and
prints the routing decision. There is no GPIO, weather client, or cloud call
yet.

This is intentionally a demo, not a production model.

## Train (Python)

From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python examples/esp32_demo/fake_esp32.py --train --device cpu
python examples/esp32_demo/fake_esp32.py --once "turn on the kitchen light"
```

`--train` expands the hand-written JSONL files with held-out templates, writes
`runs/esp32_demo/command-router.pt`, evaluates the generated test split, and
packs `firmware/main/weights.bin` for the ESP32 app.

## Classify on ESP32

The firmware is a standard ESP-IDF project. It embeds the packed TinyScorer
weights and runs the same one-pass option head in C. No extra tensor library
is required: the model is byte embeddings, layer-norm, and a few matrix
multiplies.

```sh
source /home/david/esp/idf/export.sh   # or: source $IDF_PATH/export.sh
python examples/esp32_demo/export_firmware.py
cd examples/esp32_demo/firmware
idf.py set-target esp32
idf.py build
idf.py -p PORT flash monitor
```

On boot the serial console classifies three sample phrases, then reads more
lines from UART. You should see the same GPIO / weather / cloud prints as the
Python router.
