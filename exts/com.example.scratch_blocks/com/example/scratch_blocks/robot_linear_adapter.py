from __future__ import annotations

from .model import validate_linear_parameters


def read_root_link(yaml_path: str) -> str:
    """The 'root_link:' entry of a Lula robot description YAML."""
    with open(yaml_path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip().startswith("root_link:"):
                return line.split(":", 1)[1].strip()
    raise ValueError(f"No root_link in {yaml_path}")


def kinematic_root_world_pose(stage, robot, yaml_path: str):
    """World pose of the scene prim that matches the URDF/Lula root link.

    Lula's model starts at root_link. robot.get_world_pose() returns the
    articulation root, which can sit further down the chain and already carry
    the URDF's root rotation (LARA 8: -90 deg about Y, applied twice -> arm on
    its side). Reading the matching prim avoids that for any robot.
    """
    import numpy as np
    from pxr import Usd, UsdGeom
    root_name = read_root_link(yaml_path)
    robot_root = str(robot.prim_path)
    top = robot_root.split("/")[:3]          # e.g. /World/lara8
    search_root = stage.GetPrimAtPath("/".join(top)) if len(top) == 3 else stage.GetPseudoRoot()
    matches = [p for p in Usd.PrimRange(search_root) if p.GetName() == root_name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one prim named '{root_name}' under {search_root.GetPath()}, "
                           f"found {[str(p.GetPath()) for p in matches]}")
    matrix = UsdGeom.Xformable(matches[0]).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    position = np.array(matrix.ExtractTranslation(), dtype=float)
    q = matrix.ExtractRotationQuat()
    orientation = np.array([q.GetReal(), *q.GetImaginary()], dtype=float)  # w, x, y, z
    return position, orientation


def plan_from_block(robot, parameters: dict, profile: dict | None = None):
    """Resolve robot configuration and plan; never command a movement.

    Targets are stage-world XYZ positions for absolute moves, offsets for relative moves.
    Empty block configuration uses the explicitly selected INIT profile.
    Joint-name checks cannot prove kinematic equivalence of robot models.
    """
    import math
    import carb
    import omni.usd
    from pxr import UsdGeom
    from .robot_linear import plan_linear

    from .robot_profiles import load_profile, validate_profile_robot, resolve_kinematics
    value = validate_linear_parameters(parameters)
    if profile is None:
        # Compatibility for earlier direct Panda tests only.
        profile = load_profile("franka")
    validate_profile_robot(profile, robot)
    config = resolve_kinematics(profile, value)
    frame = config["end_effector_frame"]
    descriptor = config["robot_description_path"]
    urdf = config["urdf_path"]
    carb.log_info(f"[Scratch Blocks] Linear configuration from profile={profile['name']}; frame={frame}")

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage loaded.")
    if not math.isclose(float(UsdGeom.GetStageMetersPerUnit(stage)), 1.0, abs_tol=1e-9):
        raise ValueError("Linear planning prototype requires stage units of 1 metre.")

    base_pose = kinematic_root_world_pose(stage, robot, descriptor)
    is_absolute = value.get("coordinate_mode") == "absolute"
    return plan_linear(
        robot,
        distance=value["distance"],
        steps=int(value["steps"]),
        axis=value["axis"],
        displacement=(value["x"], value["y"], value["z"]),
        target_position=(value["x"], value["y"], value["z"]) if is_absolute else None,
        robot_description_path=descriptor or None,
        urdf_path=urdf or None,
        end_effector_frame=frame,
        robot_base_pose=base_pose,
    )