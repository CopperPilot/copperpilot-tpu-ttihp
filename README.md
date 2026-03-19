![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)

# CopperPilot TPU - Tiny Tapeout Project `tt_um_copperpilot_tpu`

This repository contains the **CopperPilot TPU** designed for **Tiny Tapeout**.

- Documentation: [docs/info.md](docs/info.md)

## What this design is

The `tt_um_copperpilot_tpu` module implements a small **2x2 systolic TPU** for **signed 8-bit 2x2 matrix multiplication**, producing streamed output bytes representing the **16-bit accumulated results**.

Control is provided via `uio_in`:

- `uio_in[0]`: `load_en` (load matrix elements from `ui_in`)
- `uio_in[1]`: `transpose` (treat the second operand as transposed)
- `uio_in[2]`: `activation` (apply ReLU activation)

Outputs are streamed on:

- `uo_out[7:0]`: 8-bit chunks of the 2x2 output matrix results
- `uio_out[7]`: `done` flag for the output stream

## How to test

From the repo root:

```sh
cd test
make -B
```

This runs the cocotb block tests and the full-chip TPU north-star regression driven by `test/test.py`.
