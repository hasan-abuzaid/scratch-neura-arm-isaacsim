"""Mount any gripper on any arm in the open stage (simulation must be stopped).

How it works (no copying, no re-parenting, fully reversible):
* The gripper stays where it is in the stage tree (e.g. /World/onrobot_2fg7).
* Its root fixed joint (world -> gripper base) is re-anchored to the arm's last
  link at the tool-flange frame, and its own articulation root is removed.
  PhysX then builds ONE articulation: arm DOFs + gripper DOFs.
* The gripper prim is moved so it already sits on the flange (no snap on Play).
* Everything that was changed is recorded on the gripper prim, so unmount()
  restores the exact previous state.

Only pxr is used here, so the logic is testable outside Isaac Sim.
"""
from __future__ import annotations

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

ROOT_SCHEMAS = ("PhysicsArticulationRootAPI", "NewtonArticulationRootAPI", "PhysxArticulationAPI")
END_EFFECTOR_TYPES = ("end effector", "end_effector", "endeffector", "gripper")
FLANGE_NAMES = ("tool0", "flange", "tool_flange", "ee_link", "end_effector", "tcp_flange")

# Bookkeeping written on the gripper prim while it is mounted.
A_MOUNTED_ON = "scratchBlocks:mountedOn"          # arm robot path
A_JOINT = "scratchBlocks:mountJoint"              # path of the joint we re-anchored
A_CREATED_JOINT = "scratchBlocks:createdJoint"    # True if we created that joint
A_ORIG_XFORM = "scratchBlocks:originalLocalXform" # gripper prim local transform
A_ORIG_BODY0 = "scratchBlocks:originalBody0"
A_ORIG_POS0 = "scratchBlocks:originalLocalPos0"
A_ORIG_ROT0 = "scratchBlocks:originalLocalRot0"
A_REMOVED = "scratchBlocks:removedSchemas"        # "primPath|schema" entries
A_ORIG_OPS = "scratchBlocks:originalXformOpOrder"
MOUNT_OP_SUFFIX = "scratchMount"


# ---------------------------------------------------------------- helpers
def _all_prims(root):
    return Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))


def _inside(path: str, parent: str) -> bool:
    return path == parent or path.startswith(parent.rstrip("/") + "/")


def schemas_of(prim) -> list[str]:
    """Applied API schemas, including ones whose plugin is not registered."""
    names = list(prim.GetAppliedSchemas())
    listop = prim.GetMetadata("apiSchemas")
    if listop:
        for name in listop.GetAddedOrExplicitItems():
            if name not in names and name not in listop.deletedItems:
                names.append(name)
    return names


def is_articulation_root(prim) -> bool:
    return "PhysicsArticulationRootAPI" in schemas_of(prim) or prim.HasAPI(UsdPhysics.ArticulationRootAPI)


def is_end_effector(prim) -> bool:
    if prim.IsPseudoRoot():
        return False
    attr = prim.GetAttribute("isaac:robotType")
    value = attr.Get() if attr and attr.HasValue() else None
    return bool(value) and str(value).strip().lower() in END_EFFECTOR_TYPES


def is_mounted(prim) -> bool:
    if prim.IsPseudoRoot():
        return False
    attr = prim.GetAttribute(A_MOUNTED_ON)
    return bool(attr and attr.HasAuthoredValue() and attr.Get())


def mounted_on(prim) -> str:
    return str(prim.GetAttribute(A_MOUNTED_ON).Get()) if is_mounted(prim) else ""


def joint_names_under(prim) -> set[str]:
    return {p.GetName() for p in _all_prims(prim) if p.IsA(UsdPhysics.Joint)}


def _local_pose(matrix: Gf.Matrix4d) -> tuple[Gf.Vec3f, Gf.Quatf]:
    m = Gf.Matrix4d(matrix)
    m.Orthonormalize(False)
    q = m.ExtractRotationQuat()
    t = matrix.ExtractTranslation()
    return Gf.Vec3f(*t), Gf.Quatf(q.GetReal(), *q.GetImaginary())


def _pose_matrix(pos, rot) -> Gf.Matrix4d:
    m = Gf.Matrix4d(1.0)
    m.SetTransform(Gf.Rotation(Gf.Quatd(rot.GetReal(), Gf.Vec3d(*rot.GetImaginary()))), Gf.Vec3d(*pos))
    return m


def _world(cache: UsdGeom.XformCache, prim) -> Gf.Matrix4d:
    return cache.GetLocalToWorldTransform(prim)


def _set_local_xform(prim, matrix: Gf.Matrix4d) -> None:
    """Write a pose with dedicated ops, so referenced op types never clash."""
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    transform = Gf.Transform(matrix)
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, MOUNT_OP_SUFFIX).Set(transform.GetTranslation())
    q = transform.GetRotation().GetQuat()
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, MOUNT_OP_SUFFIX).Set(Gf.Quatd(q.GetReal(), q.GetImaginary()))
    scale = transform.GetScale()
    if any(abs(s - 1.0) > 1e-9 for s in scale):
        xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, MOUNT_OP_SUFFIX).Set(scale)


# ---------------------------------------------------------------- discovery
def find_grippers(stage, gripper_profiles: dict | None = None) -> list[str]:
    """Gripper prims in the stage: Isaac 'End Effector' robots, mounted grippers,
    and articulations whose joints match a configured gripper profile."""
    gripper_profiles = gripper_profiles or {}
    signatures = [set(g.get("joint_names", [])) for g in gripper_profiles.values() if g.get("joint_names")]
    found: list[str] = []
    for prim in _all_prims(stage.GetPseudoRoot()):
        path = str(prim.GetPath())
        if any(_inside(path, f) for f in found):
            continue
        if is_end_effector(prim) or is_mounted(prim):
            found.append(path)
        elif is_articulation_root(prim) and signatures:
            owner = prim.GetParent() if prim.IsA(UsdPhysics.Joint) else prim
            names = joint_names_under(owner)
            if any(sig <= names for sig in signatures):
                found.append(str(owner.GetPath()))
    return found


def match_gripper_profile(stage, gripper_path: str, gripper_profiles: dict) -> str | None:
    names = joint_names_under(stage.GetPrimAtPath(gripper_path))
    best, best_count = None, 0
    for name, data in gripper_profiles.items():
        joints = set(data.get("joint_names", []))
        if joints and joints <= names and len(joints) > best_count:
            best, best_count = name, len(joints)
    return best


def find_tool_flange(stage, arm_path: str, arm_profile: dict | None = None) -> tuple[str, str]:
    """(link body path, flange frame path) at the end of the arm."""
    arm = stage.GetPrimAtPath(arm_path)
    if not arm.IsValid():
        raise ValueError(f"Arm {arm_path} not found")
    prims = list(_all_prims(arm))

    def by_name(name: str):
        hits = [p for p in prims if p.GetName() == name]
        if len(hits) != 1:
            raise ValueError(f"Flange '{name}' found {len(hits)} times under {arm_path}")
        return hits[0]

    def body_of(prim):
        p = prim
        while p and p.IsValid() and _inside(str(p.GetPath()), arm_path):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                return p
            p = p.GetParent()
        return None

    config = (arm_profile or {}).get("tool_flange") or {}
    if config.get("frame") or config.get("link"):
        frame = by_name(config.get("frame") or config["link"])
        link = by_name(config["link"]) if config.get("link") else body_of(frame)
        if link is None or not link.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"Flange link of {arm_path} is not a rigid body")
        return str(link.GetPath()), str(frame.GetPath())

    # Automatic: the last link of the chain (body1 of a joint, body0 of none).
    parents, children = set(), set()
    for p in prims:
        if p.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(p)
            parents.update(str(t) for t in j.GetBody0Rel().GetTargets())
            children.update(str(t) for t in j.GetBody1Rel().GetTargets())
    leaves = sorted(c for c in children - parents
                    if stage.GetPrimAtPath(c).HasAPI(UsdPhysics.RigidBodyAPI))
    # Prefer a link that carries a flange-like frame.
    flanged = []
    for leaf in leaves:
        leaf_prim = stage.GetPrimAtPath(leaf)
        for p in _all_prims(leaf_prim):
            name = p.GetName().lower()
            if p != leaf_prim and not p.HasAPI(UsdPhysics.RigidBodyAPI) and (
                    name in FLANGE_NAMES or name.endswith("flange") or name.endswith("tool0")):
                flanged.append((leaf, str(p.GetPath())))
                break
    if len(flanged) == 1:
        return flanged[0]
    if len(leaves) == 1:
        return leaves[0], leaves[0]
    raise ValueError(f"Cannot find the tool flange of {arm_path} automatically "
                     f"(end links: {leaves}). Add 'tool_flange' to its profile.")


def _find_root(stage, gripper):
    """(root joint prim or None, root body prim, prim that holds the articulation root)."""
    for p in _all_prims(gripper):
        if is_articulation_root(p):
            if p.IsA(UsdPhysics.Joint):
                j = UsdPhysics.Joint(p)
                if not p.IsA(UsdPhysics.FixedJoint) or j.GetBody0Rel().GetTargets():
                    raise ValueError(f"Gripper root joint {p.GetPath()} is not a fixed joint to the world")
                body1 = j.GetBody1Rel().GetTargets()
                if len(body1) != 1:
                    raise ValueError(f"Gripper root joint {p.GetPath()} has no single body")
                return p, stage.GetPrimAtPath(body1[0]), p
            body = p if p.HasAPI(UsdPhysics.RigidBodyAPI) else next(
                (c for c in _all_prims(p) if c.HasAPI(UsdPhysics.RigidBodyAPI)), None)
            if body is None:
                raise ValueError(f"Gripper {gripper.GetPath()} has no rigid body")
            return None, body, p
    raise ValueError(f"{gripper.GetPath()} has no articulation root: not a gripper, or already mounted")


# ---------------------------------------------------------------- mount / unmount
def unmount(stage, gripper_path: str) -> bool:
    gripper = stage.GetPrimAtPath(gripper_path)
    if not gripper.IsValid() or not is_mounted(gripper):
        return False
    joint = stage.GetPrimAtPath(str(gripper.GetAttribute(A_JOINT).Get()))
    created = bool(gripper.GetAttribute(A_CREATED_JOINT).Get())
    if created:
        stage.RemovePrim(joint.GetPath())
    elif joint.IsValid():
        j = UsdPhysics.Joint(joint)
        body0 = str(gripper.GetAttribute(A_ORIG_BODY0).Get() or "")
        j.GetBody0Rel().SetTargets([Sdf.Path(body0)] if body0 else [])
        j.CreateLocalPos0Attr().Set(gripper.GetAttribute(A_ORIG_POS0).Get())
        j.CreateLocalRot0Attr().Set(gripper.GetAttribute(A_ORIG_ROT0).Get())
    for entry in gripper.GetAttribute(A_REMOVED).Get() or []:
        path, schema = str(entry).split("|", 1)
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            prim.AddAppliedSchema(schema)
    _set_local_xform(gripper, gripper.GetAttribute(A_ORIG_XFORM).Get())
    for name in (A_MOUNTED_ON, A_JOINT, A_CREATED_JOINT, A_ORIG_XFORM, A_ORIG_BODY0,
                 A_ORIG_POS0, A_ORIG_ROT0, A_REMOVED, A_ORIG_OPS):
        if gripper.GetAttribute(name):
            gripper.RemoveProperty(name)
    return True


def mounted_grippers(stage, arm_path: str | None = None) -> list[str]:
    return [str(p.GetPath()) for p in _all_prims(stage.GetPseudoRoot())
            if is_mounted(p) and (arm_path is None or mounted_on(p) == arm_path)]


def mount(stage, arm_path: str, gripper_path: str, arm_profile: dict | None = None) -> dict:
    """Attach gripper_path to the tool flange of arm_path. Returns a summary."""
    gripper = stage.GetPrimAtPath(gripper_path)
    if not gripper.IsValid():
        raise ValueError(f"Gripper {gripper_path} not found")
    if _inside(gripper_path, arm_path) or _inside(arm_path, gripper_path):
        raise ValueError("Gripper and arm must be separate prims in the stage")
    # One gripper per arm, one arm per gripper.
    unmount(stage, gripper_path)
    for other in mounted_grippers(stage, arm_path):
        unmount(stage, other)

    link_path, frame_path = find_tool_flange(stage, arm_path, arm_profile)
    joint, body, root_holder = _find_root(stage, gripper)

    cache = UsdGeom.XformCache()
    t_link = _world(cache, stage.GetPrimAtPath(link_path))
    t_flange = _world(cache, stage.GetPrimAtPath(frame_path))
    t_body = _world(cache, body)
    t_gripper = _world(cache, gripper)
    t_parent = _world(cache, gripper.GetParent())
    original_local = UsdGeom.Xformable(gripper).GetLocalTransformation()

    # Gripper side of the joint (where it bolts on), relative to the gripper prim.
    if joint is not None:
        j = UsdPhysics.Joint(joint)
        frame1 = _pose_matrix(j.GetLocalPos1Attr().Get() or Gf.Vec3f(0),
                              j.GetLocalRot1Attr().Get() or Gf.Quatf(1))
    else:
        frame1 = Gf.Matrix4d(1.0)
    mount_in_gripper = frame1 * t_body * t_gripper.GetInverse()
    # Move the gripper so its mounting frame sits exactly on the flange frame.
    new_world = mount_in_gripper.GetInverse() * t_flange
    pos0, rot0 = _local_pose(t_flange * t_link.GetInverse())

    # Record the original state before changing anything.
    Mat = Sdf.ValueTypeNames.Matrix4d
    gripper.CreateAttribute(A_ORIG_XFORM, Mat, custom=True).Set(original_local)
    created = joint is None
    if created:
        joint = UsdPhysics.FixedJoint.Define(stage, gripper.GetPath().AppendChild("scratch_mount_joint")).GetPrim()
        j = UsdPhysics.Joint(joint)
        j.CreateBody1Rel().SetTargets([body.GetPath()])
        j.CreateLocalPos1Attr().Set(Gf.Vec3f(0))
        j.CreateLocalRot1Attr().Set(Gf.Quatf(1))
    else:
        j = UsdPhysics.Joint(joint)
        old_body0 = j.GetBody0Rel().GetTargets()
        gripper.CreateAttribute(A_ORIG_BODY0, Sdf.ValueTypeNames.String, custom=True).Set(
            str(old_body0[0]) if old_body0 else "")
        gripper.CreateAttribute(A_ORIG_POS0, Sdf.ValueTypeNames.Point3f, custom=True).Set(
            j.GetLocalPos0Attr().Get() or Gf.Vec3f(0))
        gripper.CreateAttribute(A_ORIG_ROT0, Sdf.ValueTypeNames.Quatf, custom=True).Set(
            j.GetLocalRot0Attr().Get() or Gf.Quatf(1))
    removed = [f"{root_holder.GetPath()}|{s}" for s in ROOT_SCHEMAS if s in schemas_of(root_holder)]

    # Apply.
    for entry in removed:
        root_holder.RemoveAppliedSchema(entry.split("|", 1)[1])
    j.CreateBody0Rel().SetTargets([Sdf.Path(link_path)])
    j.CreateLocalPos0Attr().Set(pos0)
    j.CreateLocalRot0Attr().Set(rot0)
    _set_local_xform(gripper, new_world * t_parent.GetInverse())

    gripper.CreateAttribute(A_MOUNTED_ON, Sdf.ValueTypeNames.String, custom=True).Set(arm_path)
    gripper.CreateAttribute(A_JOINT, Sdf.ValueTypeNames.String, custom=True).Set(str(joint.GetPath()))
    gripper.CreateAttribute(A_CREATED_JOINT, Sdf.ValueTypeNames.Bool, custom=True).Set(created)
    gripper.CreateAttribute(A_REMOVED, Sdf.ValueTypeNames.StringArray, custom=True).Set(removed)
    return {"arm": arm_path, "gripper": gripper_path, "link": link_path,
            "flange": frame_path, "joint": str(joint.GetPath())}


def check_mounted(stage, arm_path: str, gripper_path: str) -> None:
    gripper = stage.GetPrimAtPath(gripper_path)
    if not gripper.IsValid():
        raise RuntimeError(f"Gripper {gripper_path} not found")
    if mounted_on(gripper) != arm_path:
        raise RuntimeError(f"Gripper {gripper_path} is not mounted on {arm_path}: "
                           "Stop, then select the gripper again in INIT GRIPPER")
