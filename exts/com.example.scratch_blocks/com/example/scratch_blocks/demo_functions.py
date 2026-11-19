"""Readable demonstration of the six reusable functions.

Call run_demo from an Isaac Sim async context with simulation already playing.
The function is not executed on import. Requires an empty configured gripper
and a reachable +1 cm world-X target. Physical execution still needs testing.
"""
from . import robot_control as robot_actions


async def run_demo(robot_path, profile_name, joint_name, target_degrees=15.0):
    robot, profile = robot_actions.initialize_configured_robot({
        "robot_path": robot_path, "profile": profile_name,
    })
    properties = robot_actions.get_robot_properties(robot)
    print("Robot properties:", properties)
    await robot_actions.move_joint(robot, profile, joint_name, target_degrees)
    await robot_actions.wait(2.0)
    await robot_actions.move_linear(robot, profile, {
        "axis": "x", "distance": 0.01, "steps": 5, "duration": 2.0,
        "robot_description_path": "", "urdf_path": "", "end_effector_frame": "",
    })
    await robot_actions.set_gripper(robot, profile, "open")
    await robot_actions.wait(1.0)
    await robot_actions.set_gripper(robot, profile, "close")
