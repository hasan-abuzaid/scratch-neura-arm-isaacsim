
from __future__ import annotations

import asyncio
import math
import time

import numpy as np
import omni.timeline

from isaacsim.core.utils.types import ArticulationAction


def move_joint(
    robot,
    joint_name: str,
    target_degrees: float,
):
    """
    Sendet ein Positionsziel an ein Rotationsgelenk.
    Wartet nicht auf den Abschluss der Bewegung.
    """

    timeline = omni.timeline.get_timeline_interface()

    if not timeline.is_playing():
        raise RuntimeError(
            "Bitte zuerst die Simulation starten."
        )

    names = list(robot.dof_names)

    if joint_name not in names:
        raise ValueError(
            f"Gelenk '{joint_name}' nicht gefunden. "
            f"Verfuegbare Gelenke: {names}"
        )

    joint_index = names.index(joint_name)
    target_degrees = float(target_degrees)

    if not math.isfinite(target_degrees):
        raise ValueError("Zielwinkel muss endlich sein.")

    target_radians = math.radians(target_degrees)

    view = robot._articulation_view

    if view is None:
        raise RuntimeError(
            "Articulation-View nicht initialisiert."
        )

    limits = np.asarray(view.get_dof_limits())

    if limits.ndim == 3:
        limits = limits[0]

    lower = float(limits[joint_index, 0])
    upper = float(limits[joint_index, 1])

    if not lower <= target_radians <= upper:
        raise ValueError(
            f"Ziel {target_degrees:.2f} Grad liegt "
            f"ausserhalb der Gelenkgrenzen."
        )

    action = ArticulationAction(
        joint_positions=np.array(
            [target_radians], dtype=np.float64
        ),
        joint_indices=np.array(
            [joint_index], dtype=np.int32
        ),
    )

    robot.apply_action(action)

    print(
        f"[Robot] Move Joint: {joint_name} "
        f"-> {target_degrees:.2f} Grad"
    )

    return target_radians


async def wait_for_joint_target(
    robot,
    joint_name: str,
    target_degrees: float,
    tolerance_degrees: float = 0.5,
    timeout_seconds: float = 10.0,
):
    """
    Wartet asynchron, bis ein Rotationsgelenk
    seine Zielposition erreicht hat.

    Gibt die gemessene Position in Grad zurueck.
    """

    names = list(robot.dof_names)

    if joint_name not in names:
        raise ValueError(
            f"Gelenk '{joint_name}' nicht gefunden."
        )

    joint_index = names.index(joint_name)

    target_degrees = float(target_degrees)
    tolerance_degrees = float(tolerance_degrees)
    timeout_seconds = float(timeout_seconds)

    if not math.isfinite(target_degrees):
        raise ValueError("Ungueltiger Zielwinkel.")

    if not math.isfinite(tolerance_degrees) or tolerance_degrees <= 0:
        raise ValueError("Toleranz muss positiv sein.")

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Zeitlimit muss positiv sein.")

    timeline = omni.timeline.get_timeline_interface()

    start_time = time.monotonic()

    while True:
        if not timeline.is_playing():
            raise RuntimeError(
                "Simulation wurde angehalten."
            )

        positions = robot.get_joint_positions()

        if positions is None:
            raise RuntimeError(
                "Gelenkpositionen nicht verfuegbar."
            )

        current_radians = float(
            np.asarray(positions).reshape(-1)[joint_index]
        )

        current_degrees = math.degrees(current_radians)

        difference = abs(
            current_degrees - target_degrees
        )

        if difference <= tolerance_degrees:
            print(
                f"[Robot] Ziel erreicht: {joint_name} "
                f"= {current_degrees:.2f} Grad "
                f"(Abweichung: {difference:.3f} Grad)"
            )

            return current_degrees

        if time.monotonic() - start_time >= timeout_seconds:
            raise TimeoutError(
                f"Ziel nicht erreicht: {joint_name}. "
                f"Aktuell: {current_degrees:.2f} Grad, "
                f"Ziel: {target_degrees:.2f} Grad."
            )

        await asyncio.sleep(0.05)


async def move_all_joints(robot, targets, speed_degrees_per_second=None):
    """Validate all targets before sending one action; angles are absolute degrees.

    This commands joint targets together, without Cartesian path planning or
    synchronized arrival guarantees. Existing drive settings determine motion.
    """
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        raise RuntimeError("Bitte zuerst die Simulation starten.")
    names = list(robot.dof_names)
    if not targets:
        raise ValueError("No joint targets")
    limits = np.asarray(robot._articulation_view.get_dof_limits())
    if limits.ndim == 3:
        limits = limits[0]
    indices, radians = [], []
    for name, degrees in targets.items():
        if name not in names:
            raise ValueError(f"Unknown joint: {name}")
        index = names.index(name)
        value = math.radians(float(degrees))
        if not math.isfinite(value) or not limits[index, 0] <= value <= limits[index, 1]:
            raise ValueError(f"Invalid target/outside limits: {name}={degrees}")
        indices.append(index)
        radians.append(value)
    indices = np.asarray(indices, dtype=np.int32)
    radians = np.asarray(radians, dtype=np.float64)
    if speed_degrees_per_second is None:
        robot.apply_action(ArticulationAction(joint_positions=radians, joint_indices=indices))
    else:
        await _execute_joint_ramp(robot, indices, radians, speed_degrees_per_second)
    started = time.monotonic()
    while True:
        if not timeline.is_playing():
            raise RuntimeError("Simulation wurde angehalten.")
        positions = robot.get_joint_positions()
        if positions is None:
            raise RuntimeError("Joint positions unavailable")
        actual = np.asarray(positions).reshape(-1)[indices]
        if not np.all(np.isfinite(actual)):
            raise RuntimeError("Invalid joint readback")
        errors = np.abs(np.degrees(actual - radians))
        if np.all(errors <= 0.5):
            return dict(zip(targets, np.degrees(actual).tolist()))
        if time.monotonic() - started >= 10.0:
            worst = int(np.argmax(errors))
            selected_name = names[int(indices[worst])]
            desired = float(np.degrees(radians[worst]))
            measured = float(np.degrees(actual[worst]))
            raise TimeoutError(
                f"MOVE ALL: {selected_name} target {desired:.1f} deg, "
                f"actual {measured:.1f} deg, error {float(errors[worst]):.3f} deg "
                "after 10 s; check drives, simulation and joint limits"
            )
        await asyncio.sleep(0.05)


async def _execute_joint_ramp(robot, indices, target, speed_degrees_per_second):
    """Smooth joint target ramp in simulation time; not a measured speed limit.

    Cubic smoothstep starts/ends at zero target velocity. Duration includes the
    factor 1.5 so peak target velocity of every selected joint is <= speed.
    Physical tracking and actual speed depend on drives and must be evaluated.
    """
    import omni.kit.app
    speed = float(speed_degrees_per_second)
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("Speed must be finite and positive (deg/s)")
    positions = robot.get_joint_positions()
    if positions is None:
        raise RuntimeError("Joint positions unavailable")
    initial = np.asarray(positions, dtype=np.float64).reshape(-1)[indices].copy()
    if not np.all(np.isfinite(initial)):
        raise RuntimeError("Invalid starting joint positions")
    delta = target - initial
    duration = 1.5 * float(np.max(np.abs(np.degrees(delta)))) / speed
    timeline = omni.timeline.get_timeline_interface()
    app = omni.kit.app.get_app()
    import omni.physx
    elapsed = 0.0
    last_progress = time.monotonic()
    invalid_step = False

    def on_physics_step(dt):
        nonlocal elapsed, last_progress, invalid_step
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0:
            invalid_step = True
            return
        elapsed += dt
        last_progress = time.monotonic()

    subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(on_physics_step)
    try:
        while True:
            if not timeline.is_playing():
                raise RuntimeError("Simulation wurde angehalten.")
            if invalid_step:
                raise RuntimeError("Invalid physics step duration")
            if time.monotonic() - last_progress > 10.0:
                raise TimeoutError("No physics steps received; check simulation Play")
            fraction = 1.0 if duration <= 1e-12 else min(elapsed / duration, 1.0)
            blend = fraction * fraction * (3.0 - 2.0 * fraction)
            robot.apply_action(ArticulationAction(
                joint_positions=initial + blend * delta, joint_indices=indices))
            if fraction >= 1.0:
                break
            await app.next_update_async()
    except BaseException:
        # On cancellation/error, stop advancing the commanded trajectory.
        # Holding current positions is best-effort, not an emergency stop.
        try:
            current = np.asarray(robot.get_joint_positions()).reshape(-1)[indices]
            if timeline.is_playing() and np.all(np.isfinite(current)):
                robot.apply_action(ArticulationAction(joint_positions=current.copy(), joint_indices=indices))
        except Exception:
            pass
        raise
    finally:
        subscription = None  # Release outside the physics callback on every exit.
    print(f"[Robot] Joint ramp: planned duration={duration:.3f} simulation seconds; "
          f"peak target speed <= {speed:g} deg/s; final readback check follows")


# ---------------------------------------------------------------------------
# ADD TO robot_motion.py  (paste at the end of the file)
# Continuous wave on one joint: q(t) = c + A*cos(2*pi*f*t)
# - first moves smoothly to the top of the swing (standing still there)
# - then waves with NO stop at the turning points
# - refuses settings faster than the joint's max speed
# ---------------------------------------------------------------------------

async def wave_joint(
    robot,
    joint_name: str,
    low_degrees: float,
    high_degrees: float,
    frequency_hz: float = 1.0,       # full swings (there and back) per second
    cycles: float = 3.0,             # 3 = ends at the top, 2.5 = ends at the bottom
    max_speed_deg_s: float = 180.0,  # LARA 8 datasheet: A5/A6 180, A3/A4 150, A1/A2 130
    approach_speed_deg_s: float = 60.0,
    sharpness: float = 0.0,          # 0 = pure sine (slow turns) ... 0.95 = quick, hand-like turns
):
    import omni.kit.app
    import omni.physx

    low, high = float(low_degrees), float(high_degrees)
    f, n = float(frequency_hz), float(cycles)
    if not (math.isfinite(low) and math.isfinite(high)) or high <= low:
        raise ValueError("WAVE: high must be larger than low")
    if not math.isfinite(f) or f <= 0:
        raise ValueError("WAVE: frequency must be positive")
    if n <= 0 or not (2 * n).is_integer():
        raise ValueError("WAVE: cycles must be a multiple of 0.5 (ends standing still)")

    s = float(sharpness)
    if not math.isfinite(s) or not 0.0 <= s <= 0.98:
        raise ValueError("WAVE: sharpness must be between 0 and 0.98")
    # Shape: asin(s*cos)/asin(s). s=0 -> sine; s->1 -> triangle (constant speed, quick turns).
    # Turns stay smooth (zero speed at the ends) for any s < 1.
    shaped = s > 1e-6
    speed_factor = (s / math.asin(s)) if shaped else 1.0

    center = (low + high) / 2.0
    amplitude = (high - low) / 2.0
    peak_speed = 2.0 * math.pi * f * amplitude * speed_factor
    if peak_speed > max_speed_deg_s:
        f_max = max_speed_deg_s / (2.0 * math.pi * amplitude * speed_factor)
        raise ValueError(f"WAVE: peak speed {peak_speed:.0f} deg/s > limit {max_speed_deg_s:g}; "
                         f"use frequency <= {f_max:.2f} Hz or a smaller swing")

    names = list(robot.dof_names)
    if joint_name not in names:
        raise ValueError(f"WAVE: unknown joint {joint_name}")
    index = names.index(joint_name)
    limits = np.asarray(robot._articulation_view.get_dof_limits())
    if limits.ndim == 3:
        limits = limits[0]
    if not (limits[index, 0] <= math.radians(low) and math.radians(high) <= limits[index, 1]):
        raise ValueError("WAVE: swing is outside the joint limits")

    # 1) Go to the top of the swing and stand still there (existing smooth ramp + settle).
    await move_all_joints(robot, {joint_name: high}, speed_degrees_per_second=approach_speed_deg_s)

    # 2) Continuous wave, timed in simulation time.
    indices = np.asarray([index], dtype=np.int32)
    duration = n / f
    timeline = omni.timeline.get_timeline_interface()
    app = omni.kit.app.get_app()
    elapsed = 0.0

    def on_physics_step(dt):
        nonlocal elapsed
        elapsed += float(dt)

    subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(on_physics_step)
    try:
        while True:
            if not timeline.is_playing():
                raise RuntimeError("Simulation wurde angehalten.")
            t = min(elapsed, duration)
            x = math.cos(2.0 * math.pi * f * t)
            q = center + amplitude * ((math.asin(s * x) / math.asin(s)) if shaped else x)
            robot.apply_action(ArticulationAction(
                joint_positions=np.asarray([math.radians(q)]), joint_indices=indices))
            if t >= duration:
                break
            await app.next_update_async()
    except BaseException:
        # Stop pressed / error: hold where the joint is now.
        try:
            current = np.asarray(robot.get_joint_positions()).reshape(-1)[indices]
            robot.apply_action(ArticulationAction(joint_positions=current.copy(), joint_indices=indices))
        except Exception:
            pass
        raise
    finally:
        subscription = None

    print(f"[Robot] Wave {joint_name}: {low:g}..{high:g} deg, {f:g} Hz, {n:g} cycles, sharpness {s:g}, "
          f"peak {peak_speed:.0f} deg/s, {duration:.2f} s")
