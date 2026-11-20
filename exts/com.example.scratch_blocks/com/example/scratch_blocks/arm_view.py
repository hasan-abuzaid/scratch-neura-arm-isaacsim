"""Arm-only view of an articulation that also contains a mounted gripper.

The real arm reports only its arm joints (6 for a LARA 8), while the simulated
articulation also has the gripper DOFs (e.g. finger_stroke and its mimic).
Sync, validation and the real-run mirror work in arm-joint space, so they get
this view: same interface as SingleArticulation, restricted to the arm joints
in profile order. Everything else is forwarded to the real articulation.
"""
from __future__ import annotations

import numpy as np


class ArmJointView:
    def __init__(self, articulation, arm_joint_names: list[str]):
        all_names = list(articulation.dof_names)
        missing = [n for n in arm_joint_names if n not in all_names]
        if missing:
            raise ValueError(f"Arm joints missing in the simulated robot: {missing}")
        self._art = articulation
        self._names = list(arm_joint_names)
        self._indices = np.asarray([all_names.index(n) for n in arm_joint_names], dtype=np.int32)

    # --- arm-only interface -------------------------------------------------
    @property
    def dof_names(self) -> list[str]:
        return list(self._names)

    @property
    def num_dof(self) -> int:
        return len(self._names)

    @property
    def arm_indices(self) -> np.ndarray:
        return self._indices.copy()

    def get_joint_positions(self, joint_indices=None):
        positions = np.asarray(self._art.get_joint_positions(), dtype=float).reshape(-1)[self._indices]
        return positions if joint_indices is None else positions[np.asarray(joint_indices)]

    def get_joint_velocities(self, joint_indices=None):
        velocities = np.asarray(self._art.get_joint_velocities(), dtype=float).reshape(-1)[self._indices]
        return velocities if joint_indices is None else velocities[np.asarray(joint_indices)]

    def _map(self, joint_indices):
        if joint_indices is None:
            return self._indices.copy()
        return self._indices[np.asarray(joint_indices, dtype=np.int32)]

    def set_joint_positions(self, positions, joint_indices=None):
        self._art.set_joint_positions(np.asarray(positions, dtype=float), joint_indices=self._map(joint_indices))

    def set_joint_velocities(self, velocities, joint_indices=None):
        self._art.set_joint_velocities(np.asarray(velocities, dtype=float), joint_indices=self._map(joint_indices))

    def apply_action(self, control_actions):
        from isaacsim.core.utils.types import ArticulationAction
        mapped = ArticulationAction(
            joint_positions=control_actions.joint_positions,
            joint_velocities=control_actions.joint_velocities,
            joint_efforts=control_actions.joint_efforts,
            joint_indices=self._map(control_actions.joint_indices),
        )
        self._art.apply_action(mapped)

    # --- everything else (prim_path, initialize, _articulation_view, ...) ----
    def __getattr__(self, name):
        return getattr(self._art, name)


def arm_view_for_profile(articulation, profile: dict):
    """Wrap only when the profile names the arm joints and the robot has extra DOFs."""
    arm = list(profile.get("arm_joint_names", []) or [])
    if not arm or list(articulation.dof_names) == arm:
        return articulation
    return ArmJointView(articulation, arm)
