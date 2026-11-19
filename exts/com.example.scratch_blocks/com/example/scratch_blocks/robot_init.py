
from __future__ import annotations

import numpy as np
import omni.usd
import omni.timeline

from isaacsim.core.prims import SingleArticulation


def initialize_robot(robot_path: str):
    """Initialisiert einen vorhandenen Roboter in Isaac Sim."""

    stage = omni.usd.get_context().get_stage()

    if stage is None:
        raise RuntimeError("Keine USD-Szene geladen.")

    prim = stage.GetPrimAtPath(robot_path)

    if not prim.IsValid():
        raise ValueError(
            f"Roboter unter {robot_path} nicht gefunden."
        )

    timeline = omni.timeline.get_timeline_interface()

    if not timeline.is_playing():
        raise RuntimeError(
            "Bitte zuerst die Simulation starten."
        )

    # Use the actual articulation root when it is below the selected robot prim.
    from .robot_scan import articulation_root_of
    articulation_path = articulation_root_of(robot_path)

    robot = SingleArticulation(
        prim_path=articulation_path,
        name="selected_robot"
    )

    robot.initialize()

    if robot.num_dof <= 0:
        raise RuntimeError(
            "Keine initialisierten Freiheitsgrade gefunden."
        )

    print(f"[Robot] Initialisiert: {robot_path}")

    return robot


def get_robot_properties(robot):
    """Liest allgemeine Robotereigenschaften aus."""

    num_dof = int(robot.num_dof)
    names = list(robot.dof_names)

    positions = robot.get_joint_positions()

    if positions is None:
        raise RuntimeError(
            "Gelenkpositionen konnten nicht gelesen werden."
        )

    positions = np.asarray(positions).reshape(-1)

    # Zugriff auf die interne Articulation-View.
    # Fuer den aktuellen Prototyp in Isaac Sim 6.0.1.
    view = robot._articulation_view

    if view is None:
        raise RuntimeError(
            "Articulation-View nicht initialisiert."
        )

    limits = np.asarray(view.get_dof_limits())

    # Articulation-View liefert (1, DOF, 2).
    if limits.ndim == 3 and limits.shape[0] == 1:
        limits = limits[0]

    if limits.shape != (num_dof, 2):
        raise RuntimeError(
            f"Unerwartete Gelenkgrenzen: {limits.shape}"
        )

    if len(names) != num_dof or len(positions) != num_dof:
        raise RuntimeError(
            "Gelenkdaten haben unterschiedliche Laengen."
        )

    properties = {
        "num_dof": num_dof,
        "joint_names": names,
        "joint_positions": positions.tolist(),
        "joint_limits": limits.tolist(),
    }

    print("\n=== Robot Properties ===")
    print("DOF:", num_dof)

    for index, name in enumerate(names):
        print(
            f"{index}: {name} | "
            f"Position: {positions[index]:.4f} | "
            f"Limits: {limits[index].tolist()}"
        )

    return properties
