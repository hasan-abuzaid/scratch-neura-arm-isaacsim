"""Exercise ProgramRunner.start without importing the Isaac Sim extension."""

import importlib
import math
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, call, patch


def setUpModule():
    global ProgramRunner, ProgramBlock, NeuraPyBackend, control
    package_name = "_scratch_blocks_runner_test"
    package = ModuleType(package_name)
    package.__path__ = [str(Path(__file__).resolve().parent)]
    carb = Mock(spec=["log_info", "log_error"])
    control = Mock(spec=["initialize_configured_robot", "get_robot_properties"])
    imports = patch.dict(
        sys.modules,
        {
            "carb": carb,
            package_name: package,
            f"{package_name}.robot_control": control,
        },
    )
    imports.start()
    unittest.addModuleCleanup(imports.stop)
    ProgramRunner = importlib.import_module(f"{package_name}.runner").ProgramRunner
    ProgramBlock = importlib.import_module(f"{package_name}.model").ProgramBlock
    NeuraPyBackend = importlib.import_module(
        f"{package_name}.neurapy_backend"
    ).NeuraPyBackend


class NeuraPyRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def run_blocks(self, actions, **options):
        statuses = []
        runner = ProgramRunner(statuses.append, lambda block_id: None)
        blocks = [
            ProgramBlock(str(index), kind, value, 0, 0)
            for index, (kind, value) in enumerate(actions)
        ]
        self.assertTrue(runner.start(blocks, **options))
        await runner._task
        self.assertFalse(runner.is_running)
        return statuses

    def make_target(self, dof):
        robot = Mock(
            spec=["robot_name", "dof", "get_current_joint_angles", "move_joint"]
        )
        robot.robot_name = f"Mock {dof}"
        robot.dof = dof
        robot.get_current_joint_angles.return_value = [0.125] * dof
        robot.move_joint.return_value = True
        names = tuple(f"A{axis + 1}" for axis in range(dof))
        return robot, (NeuraPyBackend(robot), names)

    async def test_joint_blocks_repeat_and_stop_for_six_and_seven_axes(self):
        for dof in (6, 7):
            with self.subTest(dof=dof):
                robot, target = self.make_target(dof)
                names = target[1]
                entries = ",".join(
                    f"{name}={10 * (names.index(name) + 1)}"
                    for name in reversed(names)
                )
                single = {"joint_name": "A2", "target_degrees": 15}
                statuses = await self.run_blocks(
                    [
                        ("initialize_robot", ""),
                        ("move_all_joints", {"targets": entries}),
                        ("repeat", 2),
                        ("move_joint", single),
                        ("end_repeat", ""),
                        ("stop_program", ""),
                        ("move_joint", single),
                    ],
                    offline_joint_target=target,
                )
                all_angles = [math.radians(10 * (i + 1)) for i in range(dof)]
                single_angles = [0.125] * dof
                single_angles[1] = math.radians(15)
                self.assertEqual(
                    robot.move_joint.call_args_list,
                    [
                        call(target_joint=[all_angles]),
                        call(target_joint=[single_angles]),
                        call(target_joint=[single_angles]),
                    ],
                )
                self.assertEqual(statuses[-1], "Completed | STOP PROGRAM")

    async def test_unknown_axis_stops_without_motion(self):
        robot, target = self.make_target(6)
        statuses = await self.run_blocks(
            [
                ("initialize_robot", ""),
                ("move_joint", {"joint_name": "A7", "target_degrees": 15}),
            ],
            offline_joint_target=target,
        )
        robot.move_joint.assert_not_called()
        self.assertEqual(statuses[-1], "Error: Unknown controller axis: A7")

    async def test_default_runner_still_uses_isaac_initialization(self):
        control.initialize_configured_robot.reset_mock()
        control.initialize_configured_robot.return_value = (
            object(),
            {"name": "franka"},
        )
        statuses = await self.run_blocks([("initialize_robot", "/panda")])
        control.initialize_configured_robot.assert_called_once_with(
            {"robot_path": "/panda", "profile": "franka"}
        )
        self.assertEqual(statuses[-1], "Completed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
