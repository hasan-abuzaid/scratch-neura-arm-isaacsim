
from __future__ import annotations

import asyncio
import time

import numpy as np
import carb
import omni.timeline

from isaacsim.core.utils.types import ArticulationAction


async def execute_linear_plan(
    robot,
    plan: dict,
    total_seconds: float = 10.0,
):
    """
    Plays the dense IK waypoints of a straight Cartesian line.

    - timed in SIMULATION time (physics steps), smooth start/stop (smoothstep)
    - between two neighbouring waypoints (~5 mm apart) joints are interpolated
    - every command carries zero velocity targets, so no leftover velocity
      target of an earlier move can pull the joints off their positions
    Only for Isaac Sim simulation.
    """
    import omni.kit.app
    import omni.physx

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        raise RuntimeError("Simulation nicht gestartet.")

    waypoints = np.asarray(plan["waypoints"], dtype=float)
    indices = np.asarray(plan["joint_indices"], dtype=np.int32)
    if waypoints.ndim != 2 or len(waypoints) < 2:
        raise ValueError("Ungueltige Wegpunkte.")
    if not np.all(np.isfinite(waypoints)):
        raise ValueError("Ungueltige Gelenkwerte.")
    if waypoints.shape[1] != len(indices):
        raise ValueError("Gelenkanzahl stimmt nicht.")
    total_seconds = float(total_seconds)
    if not np.isfinite(total_seconds) or total_seconds <= 0:
        raise ValueError("Ungueltige Dauer.")

    # Start check: still at the planned start pose (robot_control waits for
    # standstill before planning, so no velocity check here).
    current = np.asarray(robot.get_joint_positions(), dtype=float).reshape(-1)[indices]
    if not np.all(np.isfinite(current)):
        raise RuntimeError("Ungueltige aktuelle Gelenkpositionen.")
    if float(np.max(np.abs(current - waypoints[0]))) >= 0.01:
        raise RuntimeError("Startposition stimmt nicht mehr. Bitte Bahn neu berechnen.")

    sections = len(waypoints) - 1
    zero_velocity = np.zeros(len(indices))
    print("=== MOVE LINEAR AUSFUEHRUNG ===")
    print(f"Startkontrolle erfolgreich. {len(waypoints)} Bahnpunkte, {total_seconds:.2f} s")

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
            fraction = min(elapsed / total_seconds, 1.0)
            blend = fraction * fraction * (3.0 - 2.0 * fraction)   # smooth start/stop
            position = blend * sections
            k = min(int(position), sections - 1)
            alpha = position - k
            target_joints = (1.0 - alpha) * waypoints[k] + alpha * waypoints[k + 1]
            robot.apply_action(ArticulationAction(
                joint_positions=target_joints, joint_velocities=zero_velocity, joint_indices=indices))
            if fraction >= 1.0:
                break
            await app.next_update_async()
    except BaseException:
        if timeline.is_playing():
            try:
                hold = np.asarray(robot.get_joint_positions(), dtype=float).reshape(-1)[indices]
                if np.all(np.isfinite(hold)):
                    robot.apply_action(ArticulationAction(
                        joint_positions=hold.copy(), joint_velocities=zero_velocity, joint_indices=indices))
            except Exception:
                pass
        print("Bewegung abgebrochen; keine weiteren Bahnziele.")
        raise
    finally:
        subscription = None

    try:
        # Letztes Ziel noch einmal halten
        final_target = waypoints[-1]

        robot.apply_action(
            ArticulationAction(
                joint_positions=final_target.copy(),
                joint_velocities=zero_velocity,
                joint_indices=indices
            )
        )

        # Continue checking timeline while the final setpoint settles.
        for _ in range(6):
            await asyncio.sleep(0.05)
            if not timeline.is_playing():
                raise RuntimeError("Simulation wurde angehalten.")

        final_joints = np.asarray(
            robot.get_joint_positions()
        ).reshape(-1)[indices]

        if not np.all(np.isfinite(final_joints)):
            raise RuntimeError("Ungueltige Gelenkrueckmeldung am Ziel.")

        final_error = float(
            np.max(
                np.abs(final_joints - final_target)
            )
        )

        # Read-only diagnostics: no additional actions or waiting periods.
        names = list(robot.dof_names)
        carb.log_info(
            "[Scratch Blocks][LINEAR DIAG] START | "
            "joint values in native articulation units (Panda arm: rad); "
            "error=end-target; drift=end-start; planned=target-start"
        )
        for column, dof_index in enumerate(indices):
            name = names[int(dof_index)]
            start_value = float(current[column])
            target_value = float(final_target[column])
            end_value = float(final_joints[column])
            plan_start_value = float(waypoints[0, column])
            carb.log_info(
                f"[Scratch Blocks][LINEAR DIAG] {name}: "
                f"start={start_value:.9f}; "
                f"plan_start={plan_start_value:.9f}; "
                f"target={target_value:.9f}; end={end_value:.9f}; "
                f"error={end_value - target_value:+.9f}; "
                f"drift={end_value - start_value:+.9f}; "
                f"planned={target_value - start_value:+.9f}"
            )
        carb.log_info("[Scratch Blocks][LINEAR DIAG] END")

        print("=== LINEAR DIAG (deg): target | end | error ===")
        for column, dof_index in enumerate(indices):
            t = np.degrees(final_target[column]); e = np.degrees(final_joints[column])
            print(f"  {names[int(dof_index)]}: {t:9.3f} | {e:9.3f} | {e - t:+7.3f}")

        print("=== AUSFUEHRUNG BEENDET ===")
        print(
            "Maximale Gelenkabweichung am Ziel:",
            round(final_error, 5),
            "rad"
        )

        return {
            "final_joints": final_joints,
            "final_joint_error_rad": final_error,
        }

    except (asyncio.CancelledError, Exception):
        # Stop further trajectory targets and request a hold at the last
        # measured joint position. This is not an instantaneous physical stop.
        if timeline.is_playing():
            try:
                hold = np.asarray(robot.get_joint_positions(), dtype=float).reshape(-1)[indices]
                if np.all(np.isfinite(hold)):
                    robot.apply_action(ArticulationAction(
                        joint_positions=hold.copy(), joint_indices=indices
                    ))
            except Exception:
                pass  # Preserve the original exception/status.
        print("Bewegung abgebrochen; keine weiteren Bahnziele.")
        raise
