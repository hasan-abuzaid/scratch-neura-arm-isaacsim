from __future__ import annotations

from .model import validate_linear_parameters


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

    base_pose = robot.get_world_pose()
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
