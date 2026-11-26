"""Reusable robot actions below the blocks; profile data stays in robot_profiles.

No UI imports. Linear accuracy and physical grasp success are not guaranteed.
"""
from __future__ import annotations


def initialize_configured_robot(parameters):
    from .model import validate_initialize_parameters
    from .robot_init import initialize_robot
    from .robot_profiles import load_profile, validate_profile_robot
    value = validate_initialize_parameters(parameters)
    profile = load_profile(value["profile"])
    robot = initialize_robot(value["robot_path"])
    validate_profile_robot(profile, robot)
    return robot, profile


def attach_configured_gripper(robot, profile, robot_path, parameters):
    """INIT GRIPPER at run time: the gripper must already be mounted on this arm
    (done when it was chosen in the block, with the simulation stopped)."""
    import omni.usd
    from .gripper_mount import check_mounted
    from .model import validate_gripper_init_parameters
    from .robot_profiles import load_gripper_profile, with_gripper
    value = validate_gripper_init_parameters(parameters)
    stage = omni.usd.get_context().get_stage()
    check_mounted(stage, robot_path, value["gripper_path"])
    config = load_gripper_profile(value["profile"])
    names = list(robot.dof_names)
    missing = [n for n in config.get("joint_names", []) if n not in names]
    if missing:
        raise RuntimeError(f"Gripper joints {missing} are not part of {robot_path}'s articulation. "
                           "The gripper was probably mounted while playing: press Stop, then Play again.")
    return with_gripper(profile, config)


def get_robot_properties(robot):
    from .robot_init import get_robot_properties as read
    return read(robot)


async def move_joint(robot, profile, joint_name, target_degrees, speed_degrees_per_second=None):
    """Rotational arm joints in degrees; prismatic arm joints need a separate API."""
    from .robot_profiles import resolve_joint
    from .robot_motion import move_joint as send, wait_for_joint_target
    actual_name = resolve_joint(profile, joint_name)
    if speed_degrees_per_second is not None:
        from .robot_motion import move_all_joints as execute
        reached = await execute(robot, {actual_name: target_degrees},
                                speed_degrees_per_second=speed_degrees_per_second)
        return actual_name, reached[actual_name]
    send(robot, actual_name, target_degrees)
    reached = await wait_for_joint_target(robot, actual_name, target_degrees,
                                         tolerance_degrees=0.5, timeout_seconds=10.0)
    return actual_name, reached


async def wave_joint(robot, profile, joint_name, low, high, frequency, cycles,
                     approach_speed_degrees_per_second=None):
    """WAVE block in simulation: continuous wave with quick, smooth turns.

    Peak speed is checked against the LARA 8 datasheet limit of this joint.
    """
    from .robot_profiles import resolve_joint
    from .robot_motion import wave_joint as execute
    from .real_robot import max_joint_speed_deg_s
    actual_name = resolve_joint(profile, joint_name)
    await execute(robot, actual_name, low, high,
                  frequency_hz=frequency, cycles=cycles,
                  max_speed_deg_s=max_joint_speed_deg_s(actual_name),
                  approach_speed_deg_s=approach_speed_degrees_per_second or 60.0,
                  sharpness=WAVE_SHARPNESS)
    return actual_name


# 0 = pure sine (slow turns), 0.98 = almost constant speed with very quick turns.
WAVE_SHARPNESS = 0.93


async def wait(seconds):
    from .robot_wait import wait as delay
    await delay(seconds)


async def move_linear(robot, profile, parameters, on_status=lambda value: None):
    import math
    import carb
    import numpy as np
    from .robot_linear_adapter import plan_from_block
    from .robot_linear_execution import execute_linear_plan
    plan = plan_from_block(robot, parameters, profile=profile)
    sections = len(plan["waypoints"]) - 1
    carb.log_info(
        f"[Scratch Blocks] LINEAR PLAN OK | {sections} sections | profile={profile['name']}; "
        f"max planned FK residual={plan['max_position_error_m'] * 1000:.3f} mm"
    )
    carb.log_info(f"[Scratch Blocks] MOVE LINEAR world XYZ=({parameters['x']}, {parameters['y']}, {parameters['z']}); start={plan['start_position']}; target={plan['target_position']}")
    on_status(f"MOVE LINEAR: executing {sections} sections")
    result = await execute_linear_plan(robot, plan, duration=float(parameters["duration"]),
                                       sections=sections, steps_per_section=40)
    position, _ = plan["solver"].compute_forward_kinematics(plan["frame"], result["final_joints"])
    position = np.asarray(position, dtype=float).reshape(3)
    error = float(np.linalg.norm(position - plan["target_position"]))
    if not np.all(np.isfinite(position)) or not math.isfinite(error):
        raise RuntimeError("Invalid estimated linear end position")
    carb.log_info(
        f"[Scratch Blocks] Estimated end position from joint readback={position}; target={plan['target_position']}; "
        f"estimated error={error * 1000:.3f} mm; max final joint error={result['final_joint_error_rad']:.6f} rad; "
        "same kinematic model, no independent pose/path validation"
    )
    return {"plan": plan, "execution": result, "estimated_error_mm": error * 1000.0}


async def move_relative(robot, profile, parameters, on_status=lambda value: None):
    """Execute a straight Cartesian offset from the current end-effector pose."""
    value = dict(parameters)
    value["coordinate_mode"] = "relative"
    return await move_linear(robot, profile, value, on_status)


async def set_gripper(robot, profile, action):
    from .robot_gripper import set_gripper as execute
    return await execute(robot, profile, action)


async def move_all_joints(robot, profile, targets, speed_degrees_per_second=None):
    """Command all selected arm joints together, then verify their readback."""
    from .robot_motion import move_all_joints as execute
    names = list(profile.get("arm_joint_names", []))
    if not names or len(names) != len(set(names)):
        raise ValueError("No unambiguous rotational arm joints configured/detected")
    if set(targets) != set(names):
        raise ValueError(f"MOVE ALL requires exactly these joints: {names}")
    return await execute(robot, {name: targets[name] for name in names},
                         speed_degrees_per_second=speed_degrees_per_second)
