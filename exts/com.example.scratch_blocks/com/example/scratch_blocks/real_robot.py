"""Helpers for acting on the real Neura arm. No UI code, no Isaac Sim objects.

The safety rules of the "Act on real" workflow live here, in one place:
  * REAL_SPEED_PERCENT / REAL_ACCEL_PERCENT / REAL_OVERRIDE: fixed, slow motion
  * REAL_BLOCK_TYPES: blocks allowed to reach the real arm in this version
  * START_POSE_TOLERANCE_DEG: how far the real arm may be from the validated start
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

REAL_ROBOT_IP = "192.168.2.13"
REAL_SPEED_PERCENT = 10.0       # NeuraPy move_joint speed, % of max (manual p. 118)
REAL_ACCEL_PERCENT = 10.0       # NeuraPy move_joint acceleration, % of max
REAL_OVERRIDE = 0.2             # set_override: everything at 20 % (manual p. 212)
START_POSE_TOLERANCE_DEG = 0.5  # real arm must still be at the validated start pose
REAL_BLOCK_TYPES = {
    "initialize_robot", "move_joint", "move_all_joints", "wait", "set_speed",
    "print", "repeat", "end_repeat", "stop_program",
}


def prepare_neurapy_import() -> None:
    """Make pywin32's DLLs loadable inside Isaac Sim (same fix as done by hand).

    Isaac Sim ships pywin32 in exts/omni.pip.cloud/pip_prebundle, but the
    running app never registers its DLL folder, so `import win32api` fails.
    """
    try:
        import win32api  # noqa: F401  (already works: nothing to do)
        return
    except ImportError:
        pass
    for parent in Path(sys.executable).resolve().parents:
        prebundle = parent / "exts" / "omni.pip.cloud" / "pip_prebundle"
        if prebundle.is_dir():
            for sub in ("win32", os.path.join("win32", "lib")):
                path = str(prebundle / sub)
                if path not in sys.path:
                    sys.path.append(path)
            dll_dir = prebundle / "pywin32_system32"
            if dll_dir.is_dir() and hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(dll_dir))
            break
    import pywintypes  # noqa: F401  (loads pywintypes3xx.dll)
    import win32api  # noqa: F401,F811


def connect(ip: str = REAL_ROBOT_IP) -> Any:
    """Open a NeuraPy connection. Blocks on the network: call from a worker thread."""
    prepare_neurapy_import()
    from neurapy.robot import Robot
    return Robot(socket_address=ip)


def program_fingerprint(blocks: Iterable[Any]) -> str:
    """Text that changes whenever any block type or value changes."""
    return json.dumps([[block.type, block.value] for block in blocks],
                      sort_keys=True, default=str)


def unsupported_blocks(blocks: Iterable[Any]) -> list[str]:
    return sorted({block.type for block in blocks if block.type not in REAL_BLOCK_TYPES})


def max_deviation_deg(a: list[float], b: list[float]) -> float:
    """Largest joint difference in degrees; infinite if the lists don't match."""
    if len(a) != len(b):
        return math.inf
    return max((abs(math.degrees(x - y)) for x, y in zip(a, b)), default=0.0)
