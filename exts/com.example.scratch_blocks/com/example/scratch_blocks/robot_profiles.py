"""Robot-dependent data, independent of the canvas and runner."""
from __future__ import annotations
import copy
import json
import os

PROFILE_FILE = os.path.join(os.path.dirname(__file__), "robot_profiles.json")


def load_profile(name: str) -> dict:
    with open(PROFILE_FILE, encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("version") != 1 or not isinstance(data.get("profiles"), dict):
        raise ValueError("Invalid robot_profiles.json")
    if name not in data["profiles"]:
        # Compatibility: older INIT ROBOT dropdown versions wrote the robot
        # prim's basename (e.g. maira7M) as the profile. That is not a profile.
        # Use generic articulation detection instead of rejecting the robot.
        # Do not claim full motion/gripper support without configuration.
        if "generic" not in data["profiles"]:
            raise ValueError(f"Unknown robot profile '{name}'. Available: {list(data['profiles'])}")
        print(f"[Scratch Blocks] No profile named '{name}'; using generic articulation profile")
        name = "generic"
    profile = copy.deepcopy(data["profiles"][name])
    if not isinstance(profile, dict):
        raise ValueError("Robot profile must be an object")
    profile["name"] = name
    return profile


def gripper_profiles() -> dict:
    """All configured grippers ("grippers" section of robot_profiles.json)."""
    with open(PROFILE_FILE, encoding="utf-8") as stream:
        data = json.load(stream)
    grippers = data.get("grippers", {})
    if not isinstance(grippers, dict):
        raise ValueError("Invalid 'grippers' section in robot_profiles.json")
    return copy.deepcopy(grippers)


def load_gripper_profile(name: str) -> dict:
    grippers = gripper_profiles()
    if name not in grippers:
        raise ValueError(f"Unknown gripper profile '{name}'. Available: {list(grippers)} "
                         "(add the gripper to the 'grippers' section of robot_profiles.json)")
    config = grippers[name]
    config["name"] = name
    return config


def with_gripper(profile: dict, gripper_config: dict) -> dict:
    """Arm profile + a mounted gripper: the GRIPPER block then uses this gripper."""
    combined = copy.deepcopy(profile)
    combined["gripper"] = copy.deepcopy(gripper_config)
    return combined


def resolve_joint(profile: dict, name: str) -> str:
    actual = profile.get("joint_aliases", {}).get(name, name)
    if actual not in profile.get("arm_joint_names", []):
        raise ValueError(f"Joint '{name}' is not a configured rotational arm joint in profile {profile['name']}")
    return actual


def validate_profile_robot(profile: dict, robot) -> None:
    names = list(robot.dof_names)
    if profile.get("name") == "generic":
        # Detect joint type, not robot brand. Runtime data only: no JSON write.
        # A revolute joint is not automatically an arm joint (e.g. wheels).
        import omni.usd
        from pxr import Usd, UsdPhysics
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(robot.prim_path) if stage else None
        if not root or not root.IsValid():
            raise RuntimeError("Robot prim unavailable for joint type detection")
        candidates = {}
        # The articulation root may be a link; joints can be its siblings.
        # Search nearest ancestors, stopping before the whole-stage pseudo-root.
        while root and root.IsValid() and not root.IsPseudoRoot():
            candidates = {}
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if prim.IsA(UsdPhysics.Joint):
                    candidates.setdefault(prim.GetName(), []).append(prim)
            if any(name in candidates for name in names):
                break
            root = root.GetParent()
        detected = []
        for name in names:
            matches = candidates.get(name, [])
            # Reject ambiguous names and unsupported multi-axis joint types.
            if len(matches) == 1 and matches[0].IsA(UsdPhysics.RevoluteJoint):
                detected.append(name)
        profile["arm_joint_names"] = detected
        profile["joint_aliases"] = {}
        profile["label"] = "Automatically detected rotational DOFs; select joint explicitly"
        print(f"[Robot] Generic rotational joints: {detected}")
    required = list(profile.get("arm_joint_names", []))
    gripper = profile.get("gripper")
    if gripper:
        required += list(gripper.get("joint_names", []))
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"Profile {profile['name']} does not match loaded robot; missing joints: {missing}")


def resolve_kinematics(profile: dict, overrides: dict) -> dict:
    """Resolve both files together. A profile never silently becomes Franka."""
    descriptor = str(overrides.get("robot_description_path", "")).strip()
    urdf = str(overrides.get("urdf_path", "")).strip()
    frame = str(overrides.get("end_effector_frame", "")).strip()
    if bool(descriptor) != bool(urdf):
        raise ValueError("Provide both YAML and URDF paths, or leave both empty to use the INIT profile")
    config = profile.get("kinematics")
    if descriptor:
        if not frame:
            raise ValueError("An explicit YAML/URDF override requires an end-effector frame")
        config = {"provider": "files", "robot_description_path": descriptor, "urdf_path": urdf, "end_effector_frame": frame}
    if not config:
        raise ValueError(f"Profile {profile['name']} has no linear kinematics configuration")
    config = copy.deepcopy(config)
    config["end_effector_frame"] = frame or config.get("end_effector_frame", "")
    if not config["end_effector_frame"]:
        raise ValueError("End-effector frame missing in robot profile")
    if config.get("provider") == "franka_builtin":
        from isaacsim.core.utils.extensions import get_extension_path_from_name
        extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        if not extension_path:
            raise RuntimeError("Motion generation extension not available")
        folder = os.path.join(extension_path, "motion_policy_configs", "franka")
        config["robot_description_path"] = os.path.join(folder, "rmpflow", "robot_descriptor.yaml")
        config["urdf_path"] = os.path.join(folder, "lula_franka_gen.urdf")
    elif config.get("provider") != "files":
        raise ValueError("Kinematics provider must be files or franka_builtin")
    for key in ("robot_description_path", "urdf_path"):
        path = os.path.expandvars(os.path.expanduser(str(config.get(key, ""))))
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(PROFILE_FILE), path)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        config[key] = path
    return config
