"""Configured multi-joint gripper positioning; no object-grasp detection."""
from __future__ import annotations
import asyncio
import math
import time
import numpy as np
import omni.timeline
from isaacsim.core.utils.types import ArticulationAction


def gripper_parameters(robot, profile: dict, action: str):
    config = profile.get("gripper")
    if not config:
        raise ValueError(f"Profile {profile['name']} has no configured gripper")
    if action not in ("open", "close"):
        raise ValueError("GRIPPER action must be open or close")
    names = config.get("joint_names", [])
    if not names or len(set(names)) != len(names):
        raise ValueError("Gripper joint names must be non-empty and unique")
    all_names = list(robot.dof_names)
    indices = np.asarray([all_names.index(name) for name in names], dtype=np.int32)
    target = np.asarray(config["open_positions" if action == "open" else "closed_positions"], dtype=float)
    if target.shape != (len(names),) or not np.all(np.isfinite(target)):
        raise ValueError("Invalid configured gripper targets")
    tolerance = float(config.get("tolerance", 0.001))
    duration = float(config.get("duration_seconds", 1.0))
    timeout = float(config.get("timeout_seconds", 5.0))
    if any(not math.isfinite(x) or x <= 0 for x in (tolerance, duration, timeout)):
        raise ValueError("Invalid gripper tolerance/duration/timeout")
    if config.get("position_unit") == "m":
        import omni.usd
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        if stage is None or not math.isclose(float(UsdGeom.GetStageMetersPerUnit(stage)), 1.0, abs_tol=1e-9):
            raise ValueError("Metre gripper profile requires metre stage units")
    elif config.get("position_unit") != "rad":
        raise ValueError("Gripper position_unit must be m or rad")
    # Existing project API for model limits, checked before any action.
    limits = np.asarray(robot._articulation_view.get_dof_limits(), dtype=float)
    if limits.ndim == 3:
        limits = limits[0]
    # PhysX stores limits as float32: 0.019 m reads back as 0.0189999994.
    # Accept targets within a micrometre/microradian of a limit and clamp them.
    lower, upper = limits[indices, 0], limits[indices, 1]
    epsilon = 1e-6
    if not np.all((target >= lower - epsilon) & (target <= upper + epsilon)):
        raise ValueError(f"Configured gripper target {target.tolist()} is outside loaded "
                         f"robot joint limits {np.stack([lower, upper], axis=1).tolist()}")
    target = np.clip(target, lower, upper)
    return indices, target, tolerance, duration, timeout


async def set_gripper(robot, profile: dict, action: str) -> dict:
    """Open/close a configured empty gripper and await all target positions.

    A blocked gripper times out. Reaching a position does not prove a grasp.
    Stop cancels new targets and requests a hold at measured finger positions.
    """
    indices, target, tolerance, duration, timeout = gripper_parameters(robot, profile, action)
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        raise RuntimeError("Start simulation before using the gripper")
    def measured():
        values = np.asarray(robot.get_joint_positions(), dtype=float).reshape(-1)[indices]
        if not np.all(np.isfinite(values)):
            raise RuntimeError("Invalid gripper joint readback")
        return values
    start = measured().copy()
    def send(values):
        robot.apply_action(ArticulationAction(joint_positions=values.copy(), joint_indices=indices))
    try:
        for step in range(1, 41):
            if not timeline.is_playing():
                raise RuntimeError("Simulation stopped during gripper movement")
            send(start + (target - start) * step / 40.0)
            await asyncio.sleep(duration / 40.0)
        deadline = time.monotonic() + timeout
        while True:
            if not timeline.is_playing():
                raise RuntimeError("Simulation stopped during gripper movement")
            current = measured()
            error = float(np.max(np.abs(current - target)))
            if error <= tolerance:
                return {"action": action, "positions": current.tolist(), "max_error": error, "unit": profile["gripper"]["position_unit"]}
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Gripper {action} target not reached; error={error:g}; check obstruction/configuration")
            await asyncio.sleep(0.05)
    except (asyncio.CancelledError, Exception):
        if timeline.is_playing():
            try:
                send(measured())
            except Exception:
                pass
        raise
