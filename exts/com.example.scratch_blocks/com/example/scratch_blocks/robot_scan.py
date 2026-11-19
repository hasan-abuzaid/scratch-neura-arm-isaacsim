"""Find the robots in the open USD stage, like the Isaac Sim Joint Inspector.

Kept free of isaacsim imports so the canvas can always build the
INIT ROBOT menu, even before the simulation is running.
"""
from __future__ import annotations

import omni.usd

ROBOT_SCHEMA = "IsaacRobotAPI"  # Isaac Sim robot schema on the robot's asset prim


def _all_prims(root):
    from pxr import Usd
    return Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))


def _is_articulation_root(prim) -> bool:
    from pxr import UsdPhysics
    return prim.HasAPI(UsdPhysics.ArticulationRootAPI) or (
        "PhysicsArticulationRootAPI" in prim.GetAppliedSchemas()
    )


def _is_inside(path: str, parent: str) -> bool:
    return path == parent or path.startswith(parent.rstrip("/") + "/")


def find_articulation_roots() -> list[str]:
    """Return one Prim path per robot, e.g. /World/lara8 or /ur5.

    1. Prims with the Isaac robot schema (what the Joint Inspector shows).
    2. Articulation roots that are not inside such a robot prim. If the root
       is a joint (e.g. /ur5/root_joint), its parent prim is used.
    """
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return []

    robots: list[str] = []
    roots: list[str] = []
    for prim in _all_prims(stage.GetPseudoRoot()):
        if ROBOT_SCHEMA in prim.GetAppliedSchemas():
            robots.append(str(prim.GetPath()))
        elif _is_articulation_root(prim):
            if prim.IsA(UsdPhysics.Joint):
                prim = prim.GetParent()
            roots.append(str(prim.GetPath()))

    for root in roots:
        if root not in robots and not any(_is_inside(root, robot) for robot in robots):
            robots.append(root)

    print(f"[Scratch Blocks] Robots in stage: {robots}")
    return robots


def articulation_root_of(robot_path: str) -> str:
    """Path of the articulation root at or below robot_path (for SingleArticulation)."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return robot_path
    prim = stage.GetPrimAtPath(robot_path)
    if not prim or not prim.IsValid():
        return robot_path
    for child in _all_prims(prim):
        if _is_articulation_root(child):
            return str(child.GetPath())
    return robot_path


def joint_names_of(robot_path: str) -> list[str]:
    """Names of all physics joints below a robot prim (e.g. shoulder_pan_joint)."""
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return []
    root = stage.GetPrimAtPath(robot_path)
    if not root or not root.IsValid():
        return []
    return [prim.GetName() for prim in _all_prims(root) if prim.IsA(UsdPhysics.Joint)]


def revolute_joint_names_of(robot_path: str) -> list[str]:
    """Return rotational USD joints belonging to this selected robot only.

    MOVE JOINT uses degrees; fixed and prismatic joints are deliberately excluded.
    """
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return []
    root = stage.GetPrimAtPath(robot_path)
    if not root or not root.IsValid():
        return []
    names = []
    seen = set()
    for prim in _all_prims(root):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            name = prim.GetName()
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names
