
from __future__ import annotations

import asyncio
import copy
import math

import carb
from collections.abc import Callable

from .model import ProgramBlock, validate_linear_parameters, validate_initialize_parameters
from .neurapy_backend import NeuraPyBackend
from . import robot_control as control
from .robot_control import get_robot_properties


class ProgramRunner:

    def __init__(
        self,
        on_status: Callable[[str], None],
        on_active_block: Callable[[str | None], None],
        on_finished: Callable[[bool, bool], None] | None = None,
    ) -> None:

        self._on_status = on_status
        self._on_active_block = on_active_block
        # Called once per run: on_finished(success, real_robot).
        self._on_finished = on_finished

        self._task: asyncio.Task | None = None
        self._robot = None
        self._linear_plan = None
        self._profile = None

    @property
    def is_running(self) -> bool:
        return (
            self._task is not None
            and not self._task.done()
        )

    def start(
        self,
        blocks: list[ProgramBlock],
        *,
        offline_joint_target: tuple[NeuraPyBackend, tuple[str, ...]] | None = None,
    ) -> bool:

        if self.is_running:
            return False

        snapshot = [
            ProgramBlock(
                id=block.id,
                type=block.type,
                value=copy.deepcopy(block.value),
                x=block.x,
                y=block.y,
                parent_id=block.parent_id,
                next_id=block.next_id,
            )
            for block in blocks
        ]

        # Expand matched REPEAT / END REPEAT pairs into a sequential snapshot.
        # The robot is initialized only once; ordinary actions are repeated.
        try:
            snapshot = self._expand_repeats(snapshot)
        except ValueError as error:
            self._on_status(f"Repeat error: {error}")
            return False
        self._task = asyncio.ensure_future(
            self._run(snapshot, offline_joint_target)
        )

        return True

    @staticmethod
    def _expand_repeats(blocks: list[ProgramBlock]) -> list[ProgramBlock]:
        """Expand nested, paired loops while preserving block IDs for highlighting."""
        def section(start: int, nested: bool = False):
            expanded = []
            i = start
            while i < len(blocks):
                current = blocks[i]
                if current.type == "end_repeat":
                    if not nested:
                        raise ValueError("END REPEAT without REPEAT")
                    return expanded, i + 1
                if current.type == "repeat":
                    count = float(current.value)
                    if not math.isfinite(count) or not count.is_integer() or not 1 <= count <= 100:
                        raise ValueError("REPEAT expects an integer from 1 to 100")
                    inner, i = section(i + 1, nested=True)
                    if not inner:
                        raise ValueError("REPEAT is empty")
                    if len(expanded) + int(count) * len(inner) > 2000:
                        raise ValueError("Too many repeated actions (limit 2000)")
                    for _ in range(int(count)):
                        expanded.extend(inner)
                    continue
                if current.type == "initialize_robot" and nested:
                    raise ValueError("INIT ROBOT must be outside REPEAT")
                expanded.append(current)
                i += 1
            if nested:
                raise ValueError("REPEAT is missing END REPEAT")
            return expanded, i
        return section(0)[0]

    def stop(self) -> None:

        if self._task is not None:
            if not self._task.done():
                self._task.cancel()

    async def _run(
        self,
        blocks: list[ProgramBlock],
        offline_joint_target: tuple[NeuraPyBackend, tuple[str, ...]] | None = None,
    ) -> None:

        real = offline_joint_target is not None
        self._on_status("Running on REAL robot" if real else "Running")
        succeeded = False
        completed_detail = None
        joint_speed = None  # No speed block: preserve existing movement behavior.
        # Require an INIT ROBOT block in each new run.
        self._robot = None
        self._linear_plan = None
        self._profile = None

        try:
            if offline_joint_target is not None:
                if not blocks or blocks[0].type != "initialize_robot":
                    raise ValueError("Offline NeuraPy runs must start with INIT ROBOT.")

            for block in blocks:

                self._on_active_block(block.id)

                # STOP PROGRAM: end the block sequence, including enclosing REPEAT loops.
                # This is a program-flow stop, not an emergency or drive-level stop.
                if block.type == "stop_program":
                    self._on_status("STOP PROGRAM: sequence ended")
                    completed_detail = "STOP PROGRAM"
                    break

                if offline_joint_target is not None:
                    # Real robot. Timing/flow blocks run here; anything that talks
                    # to NeuraPy runs on a worker thread, because NeuraPy blocks
                    # until the move is done and Isaac Sim must keep running.
                    if block.type == "wait":
                        seconds = max(0.0, float(block.value))
                        self._on_status(f"REAL: waiting {seconds:g} s")
                        await asyncio.sleep(seconds)
                        continue
                    if block.type == "print":
                        print(f"[Scratch Blocks] {block.value}")
                        self._on_status(f"Print: {block.value}")
                        continue
                    if block.type == "set_speed":
                        self._on_status("REAL: Set Speed ignored - real moves use the fixed safe speed")
                        continue
                    completed_detail = await self._in_thread(
                        self._run_offline_joint_block, block, *offline_joint_target
                    )
                    self._on_status(completed_detail)
                    continue

                # PRINT
                if block.type == "print":

                    message = str(block.value)

                    print(
                        f"[Scratch Blocks] {message}"
                    )

                    self._on_status(
                        f"Print: {message}"
                    )

                    await asyncio.sleep(0)

                elif block.type == "set_speed":
                    from .model import validate_speed
                    joint_speed = validate_speed(block.value)
                    completed_detail = f"Joint target speed: {joint_speed:g} deg/s"
                    self._on_status(completed_detail)
                    carb.log_info(f"[Scratch Blocks] {completed_detail}; applies to MOVE JOINT and MOVE ALL")

                # WAIT
                elif block.type == "wait":

                    seconds = max(
                        0.0,
                        float(block.value)
                    )

                    self._on_status(
                        f"Waiting {seconds:g} s"
                    )

                    await control.wait(seconds)

                # INITIALIZE ROBOT
                elif block.type == "initialize_robot":

                    init_value = validate_initialize_parameters(block.value)
                    robot_path = init_value["robot_path"]
                    self._on_status(f"Initializing {robot_path} | profile={init_value['profile']}")
                    self._robot, self._profile = control.initialize_configured_robot(init_value)
                    carb.log_info(f"[Scratch Blocks] Profile selected: {self._profile['name']}")

                    print(
                        "[Scratch Blocks] "
                        f"Robot initialized: {robot_path}"
                    )

                    self._on_status(
                        f"Robot initialized: {robot_path}"
                    )

                    carb.log_info(f"[Scratch Blocks] Robot initialized: {robot_path}")
                    await asyncio.sleep(0)

                # ROBOT PROPERTIES: use the robot initialized in this run.
                elif block.type == "robot_properties":
                    if self._robot is None:
                        raise RuntimeError(
                            "ROBOT PROPERTIES requires INIT ROBOT before it."
                        )

                    self._on_status("Reading robot properties")
                    properties = get_robot_properties(self._robot)
                    dof_count = int(properties["num_dof"])
                    names = properties["joint_names"]
                    positions = properties["joint_positions"]
                    limits = properties["joint_limits"]
                    carb.log_info(
                        f"[Scratch Blocks] Robot Properties: {dof_count} DOF"
                    )
                    for index, name in enumerate(names):
                        carb.log_info(
                            f"[Scratch Blocks] {index}: {name}, "
                            f"position={positions[index]:.4f} rad, "
                            f"limits={limits[index]}"
                        )
                    completed_detail = f"Robot Properties: {dof_count} DOF"
                    self._on_status(completed_detail)
                    await asyncio.sleep(0)

                elif block.type in ("move_linear", "move_relative"):
                    if self._robot is None:
                        raise RuntimeError("MOVE LINEAR requires INIT ROBOT before it.")
                    parameters = validate_linear_parameters(block.value)
                    is_relative = block.type == "move_relative"
                    parameters["coordinate_mode"] = "relative" if is_relative else "absolute"
                    self._on_status(f"{block.type.upper()}: planning")
                    await asyncio.sleep(0)
                    action = control.move_relative if is_relative else control.move_linear
                    result = await action(self._robot, self._profile, parameters, self._on_status)
                    self._linear_plan = result["plan"]
                    completed_detail = f"{'RELATIVE' if is_relative else 'LINEAR'} EXECUTED | estimated error {result['estimated_error_mm']:.3f} mm"
                    self._on_status(completed_detail)

                elif block.type == "gripper":
                    if self._robot is None:
                        raise RuntimeError("GRIPPER requires INIT ROBOT before it.")
                    action = str(block.value).strip().lower()
                    self._on_status(f"Gripper {action}")
                    result = await control.set_gripper(self._robot, self._profile, action)
                    completed_detail = f"GRIPPER {action.upper()} | target positions reached"
                    carb.log_info(f"[Scratch Blocks] {completed_detail}; readback={result['positions']}; max error={result['max_error']:g} {result['unit']}; no grasp detection")
                    self._on_status(completed_detail)

                elif block.type == "move_all_joints":
                    if self._robot is None:
                        raise RuntimeError("MOVE ALL requires INIT ROBOT")
                    from .model import parse_joint_targets
                    targets = parse_joint_targets(block.value["targets"])
                    self._on_status("Moving all arm joints")
                    reached = await control.move_all_joints(self._robot, self._profile, targets, speed_degrees_per_second=joint_speed)
                    completed_detail = "MOVE ALL: all joint targets reached"
                    carb.log_info(f"[Scratch Blocks] {completed_detail}; degrees={reached}")
                    self._on_status(completed_detail)

                # MOVE JOINT: send target and wait until reached.
                elif block.type == "move_joint":
                    if self._robot is None:
                        raise RuntimeError("MOVE JOINT requires INIT ROBOT before it.")
                    if not isinstance(block.value, dict):
                        raise ValueError("MOVE JOINT requires joint_name and target_degrees.")

                    joint_name = str(block.value.get("joint_name", "")).strip()
                    if not joint_name:
                        raise ValueError("MOVE JOINT: joint name is empty.")
                    target_degrees = float(block.value["target_degrees"])
                    if not math.isfinite(target_degrees):
                        raise ValueError("MOVE JOINT: target angle must be finite.")

                    self._on_status(f"Moving {joint_name} to {target_degrees:g} deg")
                    actual_name, reached_degrees = await control.move_joint(
                        self._robot, self._profile, joint_name, target_degrees,
                        speed_degrees_per_second=joint_speed
                    )
                    joint_name = actual_name
                    completed_detail = (
                        f"MOVE JOINT: {joint_name} = {reached_degrees:.2f} deg"
                    )
                    carb.log_info(f"[Scratch Blocks] {completed_detail} (target reached)")
                    self._on_status(completed_detail)
                    await asyncio.sleep(0)

                else:

                    raise ValueError(
                        f"Unknown block: {block.type}"
                    )

            self._on_status(
                f"Completed | {completed_detail}" if completed_detail else "Completed"
            )
            succeeded = True

        except asyncio.CancelledError:

            self._on_status("Stopped" + (" - real robot finished its last move" if real else ""))

        except Exception as error:

            carb.log_error(f"[Scratch Blocks] Error: {error}")

            self._on_status(
                f"Error: {error}"
            )

        finally:

            if real:
                # NeuraPy: call stop() once at the end of every script (manual p. 216).
                try:
                    await asyncio.to_thread(offline_joint_target[0].finish)
                except Exception as error:
                    carb.log_error(f"[Scratch Blocks] NeuraPy finish failed: {error}")
            self._on_active_block(None)
            self._task = None
            if self._on_finished is not None:
                self._on_finished(succeeded, real)

    async def _in_thread(self, function, *args):
        """Run a blocking call on a worker thread.

        If Stop is pressed meanwhile, the running NeuraPy move is NOT interrupted
        (a thread can't be killed, and interrupting a real move from outside is
        not documented). We wait for it to finish, then stop before the next block.
        """
        future = asyncio.ensure_future(asyncio.to_thread(function, *args))
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            self._on_status("Stopping: the real robot finishes its current move first")
            try:
                await future
            except Exception:
                pass
            raise

    @staticmethod
    def _run_offline_joint_block(
        block: ProgramBlock,
        backend: NeuraPyBackend,
        axis_names: tuple[str, ...],
    ) -> str:
        """Dispatch a block to an injected test robot using explicit axis order."""
        if block.type == "initialize_robot":
            robot_name, dof = backend.initialize()
            if len(axis_names) != dof or len(set(axis_names)) != dof:
                raise ValueError(f"Expected {dof} distinct controller axis names.")
            return f"NeuraPy offline | {robot_name} | {dof} joints"

        if block.type == "move_joint":
            joint_name = str(block.value.get("joint_name", "")).strip()
            if joint_name not in axis_names:
                raise ValueError(f"Unknown controller axis: {joint_name}")
            backend.move_single_joint(
                axis_names.index(joint_name), float(block.value["target_degrees"])
            )
            return f"NeuraPy offline | MOVE JOINT {joint_name}"

        if block.type == "move_all_joints":
            from .model import parse_joint_targets
            targets = parse_joint_targets(block.value["targets"])
            if set(targets) != set(axis_names):
                raise ValueError("MOVE ALL requires one angle per controller axis.")
            backend.move_all_joints([targets[name] for name in axis_names])
            return "NeuraPy offline | MOVE ALL"

        raise ValueError(f"Block is not available on the real robot in this version: {block.type}")
