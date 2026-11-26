from __future__ import annotations

import os
import numpy as np

from isaacsim.core.utils.rotations import rot_matrix_to_quat
from isaacsim.robot_motion.motion_generation import (
    LulaKinematicsSolver,
    ArticulationKinematicsSolver,
)


POINT_SPACING_M = 0.005   # IK point every 5 mm along the straight line


def plan_linear(
    robot,
    distance: float = 0.05,
    steps: int = 5,
    axis: str = "x",
    robot_description_path: str | None = None,
    urdf_path: str | None = None,
    end_effector_frame: str = "",
    robot_base_pose=None,
    displacement=None,
    target_position=None,
):
    """
    Plant eine kartesische Gerade mit Lula IK.

    Die Funktion bewegt den Roboter NICHT.

    Parameter:
        robot: Initialisierte Isaac-Sim-Articulation
        distance: Strecke in Metern
        steps: Anzahl der Bahnabschnitte
        axis: Bewegungsachse x, y oder z
        robot_description_path: Lula-YAML-Datei
        urdf_path: Roboter-URDF
        end_effector_frame: Kinematik-Frame

    Rueckgabe:
        Dictionary mit Wegpunkten und Metadaten.
    """

    distance = float(distance)
    steps = int(steps)

    if not np.isfinite(distance):
        raise ValueError("Ungueltige Distanz.")

    if steps < 1:
        raise ValueError("steps muss mindestens 1 sein.")

    if axis not in ("x", "y", "z"):
        raise ValueError("Achse muss x, y oder z sein.")

    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    if displacement is None:
        offset = np.zeros(3, dtype=float)
        offset[axis_index] = distance
    else:
        offset = np.asarray(displacement, dtype=float).reshape(3)
    if not np.all(np.isfinite(offset)):
        raise ValueError("Invalid XYZ displacement")

    if not robot_description_path or not urdf_path:
        raise ValueError("Linear planning requires YAML and URDF from a robot profile.")

    for path in (robot_description_path, urdf_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    # Kinematiksolver erstellen
    solver = LulaKinematicsSolver(
        robot_description_path=robot_description_path,
        urdf_path=urdf_path,
    )

    # Synchronize world coordinates explicitly. The adapter supplies the
    # selected articulation pose; direct legacy callers remain compatible.
    if robot_base_pose is not None:
        base_position, base_orientation = robot_base_pose
        solver.set_robot_base_pose(base_position, base_orientation)

    if end_effector_frame not in solver.get_all_frame_names():
        raise ValueError(
            f"Unbekannter Frame: {end_effector_frame}"
        )

    articulation_solver = ArticulationKinematicsSolver(
        robot_articulation=robot,
        kinematics_solver=solver,
        end_effector_frame_name=end_effector_frame,
    )

    # Aktuelle Gelenkkonfiguration
    all_names = list(robot.dof_names)
    active_names = list(solver.get_joint_names())

    all_positions = robot.get_joint_positions()

    if all_positions is None:
        raise RuntimeError("Gelenkpositionen fehlen.")

    all_positions = np.asarray(
        all_positions, dtype=float
    ).reshape(-1)

    indices = np.array([
        all_names.index(name)
        for name in active_names
    ], dtype=np.int32)

    start_joints = all_positions[indices].copy()

    # Aktuelle Endeffektorpose
    start_position, start_rotation = (
        articulation_solver.compute_end_effector_pose()
    )

    start_position = np.asarray(
        start_position, dtype=float
    ).reshape(3)

    start_rotation = np.asarray(
        start_rotation, dtype=float
    ).reshape(3, 3)

    quaternion = rot_matrix_to_quat(start_rotation)

    # Absolute target is converted to a displacement from the measured pose.
    if target_position is not None:
        absolute_target = np.asarray(target_position, dtype=float).reshape(3)
        if not np.all(np.isfinite(absolute_target)):
            raise ValueError("Invalid absolute XYZ target")
        offset = absolute_target - start_position

    # Gelenkgrenzen aus dem Robotermodell
    view = robot._articulation_view

    if view is None:
        raise RuntimeError("Articulation-View fehlt.")

    limits = np.asarray(view.get_dof_limits())

    if limits.ndim == 3:
        limits = limits[0]

    # Dense path: one IK point every POINT_SPACING_M along the line (at least
    # the requested number of sections), so the tip is on the line everywhere,
    # not only at a few waypoints.
    requested_steps = steps
    steps = max(requested_steps, int(np.ceil(float(np.linalg.norm(offset)) / POINT_SPACING_M)))

    # Wegpunkte planen
    waypoints = [start_joints.copy()]
    previous = start_joints.copy()

    max_position_error = 0.0
    max_orientation_error = 0.0

    print("=== MOVE LINEAR PLANUNG ===")
    print("Startposition:", start_position)
    print("World displacement XYZ:", offset)

    for i in range(1, steps + 1):

        target = start_position.copy()
        target += offset * i / steps

        solution, success = (
            solver.compute_inverse_kinematics(
                frame_name=end_effector_frame,
                target_position=target,
                target_orientation=quaternion,
                warm_start=previous,
            )
        )

        if not success:
            raise RuntimeError(
                f"IK fehlgeschlagen bei Punkt {i}."
            )

        solution = np.asarray(
            solution, dtype=float
        ).reshape(-1)

        if not np.all(np.isfinite(solution)):
            raise RuntimeError(
                f"Ungueltiger Wegpunkt {i}."
            )

        joint_change = float(np.max(
            np.abs(solution - previous)
        ))

        if joint_change > 0.2:
            raise RuntimeError(
                f"Grosser Gelenksprung bei Punkt {i}."
            )

        for j, value in enumerate(solution):
            lower, upper = limits[indices[j]]

            if not lower <= value <= upper:
                raise RuntimeError(
                    f"Gelenkgrenze verletzt: {active_names[j]}"
                )

        # Vorwaertskinematik zur Kontrolle
        actual_position, actual_rotation = (
            solver.compute_forward_kinematics(
                end_effector_frame,
                solution,
            )
        )

        actual_position = np.asarray(actual_position)
        actual_rotation = np.asarray(actual_rotation)

        position_error = float(np.linalg.norm(
            actual_position - target
        ))

        relative_rotation = (
            start_rotation.T @ actual_rotation
        )

        cos_angle = np.clip(
            (np.trace(relative_rotation) - 1.0) / 2.0,
            -1.0,
            1.0,
        )

        orientation_error = float(
            np.degrees(np.arccos(cos_angle))
        )

        if position_error >= 0.002:
            raise RuntimeError(
                f"Positionsfehler bei Punkt {i}."
            )

        if orientation_error >= 0.5:
            raise RuntimeError(
                f"Orientierungsfehler bei Punkt {i}."
            )

        max_position_error = max(
            max_position_error, position_error
        )

        max_orientation_error = max(
            max_orientation_error, orientation_error
        )

        waypoints.append(solution.copy())
        previous = solution.copy()

        if i == steps or i % max(1, steps // 5) == 0:
          print(
            f"Punkt {i}/{steps}: "
            f"Fehler={position_error * 1000:.3f} mm, "
            f"Orientierung={orientation_error:.3f} Grad"
          )

    print("=== PLANUNG ERFOLGREICH ===")
    print("Roboter wurde NICHT bewegt.")

    return {
        "joint_names": active_names,
        "requested_sections": requested_steps,
        "joint_indices": indices,
        "waypoints": np.asarray(waypoints),
        "start_position": start_position,
        "target_position": target,
        "max_position_error_m": max_position_error,
        "max_orientation_error_deg": max_orientation_error,
        "frame": end_effector_frame,
        "solver": solver,
    }