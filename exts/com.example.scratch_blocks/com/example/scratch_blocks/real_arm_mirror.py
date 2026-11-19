"""Live mirror: the simulated arm in Isaac Sim follows the real Neura arm.

How it works (two workers, one shared "latest reading"):

  READER THREAD (background)            MAIN THREAD (Isaac Sim: UI + physics)
  --------------------------            --------------------------------------
  loop ~30x per second:                 every rendered frame:
    angles = robot.get_current_            take the newest reading (if any)
             joint_angles()   <-- slow,    set_joint_positions(angles)   (where joints ARE)
             waits on network              apply_action(angles)          (where motors GO)
    store it as "latest"  ---------->
                         (lock-protected)

Why two? NeuraPy calls wait for the network. On Isaac Sim's main thread that
wait would freeze the whole app 30 times per second. The reader thread does
all the waiting; the main thread only copies numbers, which is instant.
Isaac Sim objects (sim_arm) are only ever touched on the main thread.

Use (Script Editor, simulation playing, after the pywin32 setup lines):

    mirror = RealArmMirror(r, sim_arm)   # r = NeuraPy Robot, sim_arm = SingleArticulation
    mirror.start()
    ...
    print(mirror.status())
    mirror.stop()

Read-only: this file never sends a motion command to the real robot.
"""
from __future__ import annotations

import asyncio
import math
import threading
import time
from typing import Any, Awaitable, Callable

import numpy as np


class RealArmMirror:
    def __init__(self, real_robot: Any, sim_arm: Any, rate_hz: float = 30.0,
                 next_frame: Callable[[], Awaitable[None]] | None = None) -> None:
        self._robot = real_robot          # NeuraPy Robot (real arm)
        self._sim = sim_arm               # Isaac Sim SingleArticulation (3D arm)
        self._period = 1.0 / float(rate_hz)
        self._next_frame = next_frame     # injectable for tests; Isaac frame by default

        self._lock = threading.Lock()     # guards the four fields below
        self._latest: list[float] | None = None
        self._latest_time = 0.0
        self._reads = 0
        self._last_error: str | None = None

        self._running = False
        self._thread: threading.Thread | None = None
        self._task: asyncio.Future | None = None
        self._applied_time = 0.0          # timestamp of the reading last shown in Isaac

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        if self._running:
            return
        if len(self._sim.dof_names) != int(self._robot.dof):
            raise ValueError(f"Joint count differs: sim {len(self._sim.dof_names)} "
                             f"vs real {self._robot.dof}")
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, name="real-arm-reader",
                                        daemon=True)
        self._thread.start()
        self._task = asyncio.ensure_future(self._apply_loop())
        print("[Mirror] started")

    def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        print("[Mirror] stopped")

    def status(self) -> str:
        with self._lock:
            age = time.monotonic() - self._latest_time if self._latest_time else float("inf")
            latest, reads, error = self._latest, self._reads, self._last_error
        degrees = [round(math.degrees(a), 1) for a in latest] if latest else None
        state = "OK" if age < 0.5 else "STALE - no fresh reading"
        return (f"[Mirror] {state} | reads={reads} | last reading {age:.2f}s ago | "
                f"deg={degrees} | last error={error}")

    # ---------------------------------------------------- reader thread (network)
    def _read_loop(self) -> None:
        while self._running:
            started = time.monotonic()
            try:
                angles = [float(a) for a in self._robot.get_current_joint_angles()]
                if len(angles) != len(self._sim.dof_names) or not all(map(math.isfinite, angles)):
                    raise ValueError(f"bad reading: {angles}")
                with self._lock:
                    self._latest, self._latest_time = angles, time.monotonic()
                    self._reads += 1
            except Exception as error:     # keep mirroring through short network hiccups
                with self._lock:
                    self._last_error = f"{type(error).__name__}: {error}"
                time.sleep(1.0)            # don't hammer a controller that isn't answering
                continue
            # Sleep only the remainder of the period, so the rate stays ~rate_hz.
            time.sleep(max(0.0, self._period - (time.monotonic() - started)))

    # ------------------------------------------------- main thread (Isaac Sim)
    async def _apply_loop(self) -> None:
        if self._next_frame is None:
            import omni.kit.app
            app = omni.kit.app.get_app()
            next_frame = app.next_update_async
        else:
            next_frame = self._next_frame
        from isaacsim.core.utils.types import ArticulationAction
        while self._running:
            await next_frame()
            with self._lock:
                angles, stamp = self._latest, self._latest_time
            if angles is None or stamp == self._applied_time:
                continue                   # nothing new since last frame
            target = np.array(angles)
            self._sim.set_joint_positions(target)                                 # IS
            self._sim.apply_action(ArticulationAction(joint_positions=target))    # GOES
            self._applied_time = stamp
