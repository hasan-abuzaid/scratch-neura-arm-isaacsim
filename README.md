# Scratch Canvas for Isaac Sim 6.0.1 — v1.2.0

This package replaces the earlier stack-layout prototype with a coordinate-based
block canvas. Ordinary `VStack` layout is not used to position program blocks.

## Implemented

- Free placement on a 1400 × 900 canvas
- `Print` and `Wait` library blocks
- Native drag-and-drop
- SVG puzzle bodies with top notches and bottom tabs
- Proximity snapping to `START` and command blocks
- Full SVG-scale 360 × 52 pixel blocks
- Measured 34-pixel Isaac canvas pitch so rendered blocks touch directly
- Drag a connected block to move its complete downstream stack
- Drop a connected block on empty canvas to disconnect that stack
- Run/Stop with active block highlighting
- Compile validation with errors and generated instructions in the Isaac console
- Save/load the coordinate and connection model as JSON

## Install

1. Disable the old **Scratch Blocks Demo** extension.
2. Replace the old `scratch_blocks_demo` folder completely with this package.
3. Confirm the extension search path points to:

   ```text
   <location>/scratch_blocks_demo/exts
   ```

4. Re-enable **Scratch Blocks Demo**. Restart Isaac Sim if its Python module
   cache still shows the old interface.

## Acceptance test

1. Drag `Print text` from the library to empty canvas.
2. Drag it close to the bottom connector on `START`; it should snap.
3. Drag `Wait seconds` close to the bottom connector on `Print`; it should snap
   with its top notch interlocked and no layout gap.
4. Drag `Print`; the connected `Wait` block should move with it.
5. Drag `Wait` onto empty canvas; it should disconnect.
6. Reconnect it, then press **Run**.

Only blocks connected beneath `START` execute. Free canvas blocks do not.

## Core architecture

- `model.py`: coordinates, parent/next links, snapping, disconnection, reflow
- `extension.py`: absolute canvas, SVG rendering, drag/drop and controls
- `runner.py`: asynchronous sequential execution
- `assets/*.svg`: puzzle-shaped block bodies
