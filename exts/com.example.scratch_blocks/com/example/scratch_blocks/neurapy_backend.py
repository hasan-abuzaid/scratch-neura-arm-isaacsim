"""NeuraPy joint commands for Scratch Robot Blocks, tested offline.

Every method here BLOCKS on the network (NeuraPy waits until a move is done).
Inside Isaac Sim, call them from a worker thread, never from the main thread.
"""

from math import isfinite, radians
import time


class NeuraPyBackend:
    """Use an injected NeuraPy-compatible robot supplied by the caller.

    speed_percent / acceleration_percent: NeuraPy move_joint speed and
    acceleration in % of maximum (manual p. 118). None keeps NeuraPy's default.
    override: set_override value 0..1 applied at initialize (manual p. 212).
    """

    def __init__(self, robot, speed_percent=None, acceleration_percent=None, override=None):
        self._robot = robot
        self._dof: int | None = None
        self._speed = speed_percent
        self._acceleration = acceleration_percent
        self._override = override
        self._gripper: dict | None = None   # real-gripper config after select_gripper()

    def initialize(self) -> tuple[str, int]:
        """Check the robot can move, read name and joint count, apply the override."""
        self._dof = None
        check = getattr(self._robot, "is_robot_in_teach_mode", None)
        if check is not None and check():
            # Motion would raise WrongMode (manual p. 119). Switching modes is a
            # human decision at the pendant, so we refuse instead of switching.
            raise RuntimeError("Robot is in teach mode: switch to automatic mode at the pendant.")
        name = self._robot.robot_name
        dof = self._robot.dof
        if type(dof) is not int or dof < 1:
            raise ValueError("NeuraPy reported an invalid joint count.")
        if self._override is not None and self._robot.set_override(self._override) is False:
            raise RuntimeError("NeuraPy refused the speed override.")
        self._dof = dof
        return name, dof

    def read_joint_angles(self) -> list[float]:
        """Current joint angles in radians, controller axis order."""
        angles = [float(a) for a in self._robot.get_current_joint_angles()]
        if not all(isfinite(a) for a in angles):
            raise ValueError("NeuraPy returned invalid current joint angles.")
        return angles

    def move_all_joints(self, targets_degrees: list[float]) -> None:
        """Move to absolute joint angles in controller axis order."""
        dof = self._require_joint_count()
        if len(targets_degrees) != dof:
            raise ValueError(f"Expected {dof} joint angles.")
        current = self.read_joint_angles() if self._speed is not None else None
        self._move_joint_vector([radians(angle) for angle in targets_degrees], current)

    def move_single_joint(self, axis_index: int, target_degrees: float) -> None:
        """Replace one absolute angle using a zero-based controller axis index."""
        dof = self._require_joint_count()
        if type(axis_index) is not int or not 0 <= axis_index < dof:
            raise ValueError(f"Joint index must be between 0 and {dof - 1}.")
        current = list(self._robot.get_current_joint_angles())
        if len(current) != dof or not all(isfinite(angle) for angle in current):
            raise ValueError("NeuraPy returned invalid current joint angles.")
        targets = list(current)
        targets[axis_index] = radians(target_degrees)
        self._move_joint_vector(targets, current if self._speed is not None else None)

    def wave_single_joint(self, axis_index: int, low_degrees: float, high_degrees: float,
                          cycles: float) -> int:
        """WAVE on the real arm: ONE move_joint call through high, low, high, ...
        with enable_blending=True, so the arm does not stop at every point
        (manual: move_joint / check_joint_feasibility). Other joints stay put.
        Speed/acceleration are the fixed real settings; WAVE frequency is ignored.
        Returns the number of points sent.
        """
        dof = self._require_joint_count()
        if type(axis_index) is not int or not 0 <= axis_index < dof:
            raise ValueError(f"Joint index must be between 0 and {dof - 1}.")
        if not (isfinite(low_degrees) and isfinite(high_degrees)) or high_degrees <= low_degrees:
            raise ValueError("WAVE: high must be larger than low.")
        half_swings = int(round(2 * float(cycles)))
        if half_swings < 1:
            raise ValueError("WAVE: at least half a cycle is required.")
        current = self.read_joint_angles()
        if len(current) != dof:
            raise ValueError("NeuraPy returned the wrong number of joint angles.")
        targets = []
        for k in range(half_swings + 1):   # high, low, high, ... (same order as the simulation)
            point = list(current)
            point[axis_index] = radians(high_degrees if k % 2 == 0 else low_degrees)
            targets.append(point)
        command = {"target_joint": targets, "enable_blending": True,
                   "current_joint_angles": current}
        if self._speed is not None:
            command["speed"] = self._speed
            command["acceleration"] = (self._acceleration
                                       if self._acceleration is not None else self._speed)
        check = getattr(self._robot, "check_joint_feasibility", None)
        if check is not None:
            result = check(**command)
            if isinstance(result, (list, tuple)) and result and result[0] is False:
                raise RuntimeError("Controller says this WAVE is not feasible - nothing was moved.")
        if self._robot.move_joint(**command) is not True:
            raise RuntimeError("NeuraPy did not report successful WAVE motion.")
        return len(targets)

    def move_joint_path(self, points_rad: list[list[float]]) -> int:
        """Several joint poses (radians, controller order) as ONE move_joint call
        with enable_blending=True: the arm moves through them without stopping.
        The controller checks feasibility first; nothing moves if it says no."""
        if not points_rad:
            raise ValueError("Empty joint path.")
        dof = self._require_joint_count()
        for point in points_rad:
            if len(point) != dof or not all(isfinite(v) for v in point):
                raise ValueError("Invalid joint path point.")
        current = self.read_joint_angles()
        command = {"target_joint": [list(p) for p in points_rad], "enable_blending": True,
                   "current_joint_angles": current}
        if self._speed is not None:
            command["speed"] = self._speed
            command["acceleration"] = (self._acceleration
                                       if self._acceleration is not None else self._speed)
        check = getattr(self._robot, "check_joint_feasibility", None)
        if check is not None:
            result = check(**command)
            if isinstance(result, (list, tuple)) and result and result[0] is False:
                raise RuntimeError("Controller says this path is not feasible - nothing was moved.")
        if self._robot.move_joint(**command) is not True:
            raise RuntimeError("NeuraPy did not report a successful path motion.")
        return len(points_rad)

    def select_gripper(self, real_config: dict) -> str:
        """INIT GRIPPER on the real arm: activate the tool for this connection.

        Lab result 20 Nov 2026: grasp()/release() fail with "'Robot' object has
        no attribute 'gripper_name'" until set_tool() is called on the SAME
        connection, even when the tool is already selected on the controller.
        """
        name = str(real_config.get("tool_name", "")).strip()
        if not name:
            raise ValueError("Gripper profile has no real tool_name")
        for key in ("open_call", "close_call"):
            if not str(real_config.get(key, "")).strip():
                raise ValueError(f"Gripper profile has no real {key}")
        if self._robot.set_tool(tool_name=name) is False:
            raise RuntimeError(f"NeuraPy refused set_tool('{name}')")
        self._gripper = dict(real_config)
        return name

    def gripper(self, action: str) -> float:
        """Open or close the real gripper, then wait until it has had time to move.

        Lab result 20 Nov 2026 (2FG7): release() returned after 0.41 s, grasp()
        after 0.03 s while the fingers were still moving - so a fixed minimum
        time is waited after every command. Returns the seconds taken.
        """
        if self._gripper is None:
            raise RuntimeError("INIT GRIPPER is required before GRIPPER on the real robot.")
        if action not in ("open", "close"):
            raise ValueError("GRIPPER action must be open or close")
        call = getattr(self._robot, self._gripper[f"{action}_call"])
        settle = float(self._gripper.get("settle_seconds", 1.0))
        started = time.monotonic()
        if call() is False:
            raise RuntimeError(f"NeuraPy did not confirm gripper {action}")
        remaining = settle - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)
        return time.monotonic() - started

    def finish(self) -> None:
        """End the NeuraPy script properly (manual: call stop() once at the end)."""
        stop = getattr(self._robot, "stop", None)
        if stop is not None:
            stop()

    def _require_joint_count(self) -> int:
        if self._dof is None:
            raise RuntimeError("INIT ROBOT is required before joint motion.")
        return self._dof

    def _move_joint_vector(self, targets_radians: list[float], current=None) -> None:
        if not all(isfinite(angle) for angle in targets_radians):
            raise ValueError("Joint targets must be finite.")
        # NeuraPy expects a list of configurations, even for a single target.
        command = {"target_joint": [targets_radians]}
        if self._speed is not None:
            command["speed"] = self._speed
            command["acceleration"] = (self._acceleration
                                       if self._acceleration is not None else self._speed)
        if current is not None:
            command["current_joint_angles"] = current
        if self._robot.move_joint(**command) is not True:
            raise RuntimeError("NeuraPy did not report successful joint motion.")
