
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
    duration: float = 2.0,
    sections: int = 1,
    steps_per_section: int = 40,
):
    """
    Fuehrt einen vorberechneten Gelenkweg aus.

    Die kartesische Bahn wird angenaehert.
    Gelenkpositionen werden zur Diagnose zurueckgelesen.
    Dies ist keine unabhaengige kartesische Bahnvermessung.

    Nur fuer Isaac-Sim-Simulation vorgesehen.
    Die neue Scratch-Anbindung ist noch in der Simulation zu validieren.
    """

    timeline = omni.timeline.get_timeline_interface()

    if not timeline.is_playing():
        raise RuntimeError("Simulation nicht gestartet.")

    waypoints = np.asarray(
        plan["waypoints"], dtype=float
    )

    indices = np.asarray(
        plan["joint_indices"], dtype=np.int32
    )

    if waypoints.ndim != 2:
        raise ValueError("Ungueltige Wegpunkte.")

    if not 1 <= sections < len(waypoints):
        raise ValueError("Ungueltige Abschnittsanzahl.")

    if not np.all(np.isfinite(waypoints)):
        raise ValueError("Ungueltige Gelenkwerte.")

    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Ungueltige Dauer.")

    if steps_per_section < 1:
        raise ValueError("Ungueltige Schrittanzahl.")

    if waypoints.shape[1] != len(indices):
        raise ValueError("Gelenkanzahl stimmt nicht.")

    # Pruefen, ob der Roboter noch an der
    # Startkonfiguration steht.
    current = np.asarray(
        robot.get_joint_positions(), dtype=float
    ).reshape(-1)[indices]
    if not np.all(np.isfinite(current)):
        raise RuntimeError("Ungueltige aktuelle Gelenkpositionen.")

    start_error = float(
        np.max(np.abs(current - waypoints[0]))
    )

    if start_error >= 0.01:
        raise RuntimeError(
            "Startposition stimmt nicht mehr. "
            "Bitte Bahn neu berechnen."
        )

    velocities = np.asarray(
        robot.get_joint_velocities()
    ).reshape(-1)[indices]

    if not np.all(np.isfinite(velocities)):
        raise RuntimeError("Ungueltige Gelenkgeschwindigkeiten.")

    if np.max(np.abs(velocities)) >= 0.02:
        raise RuntimeError(
            "Roboter bewegt sich noch."
        )

    print("=== MOVE LINEAR AUSFUEHRUNG ===")
    print("Startkontrolle erfolgreich.")
    print("Abschnitte:", sections)

    total_steps = sections * steps_per_section
    total_duration = duration * sections

    start_time = time.monotonic()

    # Messwerte fuer die spaetere Evaluation
    measured_joints = []

    try:
        for step in range(1, total_steps + 1):

            if not timeline.is_playing():
                raise RuntimeError(
                    "Simulation wurde angehalten."
                )

            progress = step / steps_per_section
            segment = min(
                int(np.ceil(progress)) - 1,
                sections - 1
            )

            alpha = progress - segment

            q0 = waypoints[segment]
            q1 = waypoints[segment + 1]

            target_joints = (
                (1.0 - alpha) * q0 + alpha * q1
            )

            robot.apply_action(
                ArticulationAction(
                    joint_positions=target_joints,
                    joint_indices=indices
                )
            )

            actual_joints = np.asarray(
                robot.get_joint_positions()
            ).reshape(-1)[indices]

            measured_joints.append(
                actual_joints.copy()
            )

            next_time = (
                start_time
                + step / total_steps * total_duration
            )

            await asyncio.sleep(
                max(
                    0.0,
                    next_time - time.monotonic()
                )
            )

        # Letztes Ziel noch einmal halten
        final_target = waypoints[sections]

        robot.apply_action(
            ArticulationAction(
                joint_positions=final_target.copy(),
                joint_indices=indices
            )
        )

        # Continue checking timeline while the final setpoint settles.
        for _ in range(10):
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

        print("=== AUSFUEHRUNG BEENDET ===")
        print(
            "Maximale Gelenkabweichung am Ziel:",
            round(final_error, 5),
            "rad"
        )

        return {
            "final_joints": final_joints,
            "final_joint_error_rad": final_error,
            "measured_joints": np.asarray(
                measured_joints
            ),
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
