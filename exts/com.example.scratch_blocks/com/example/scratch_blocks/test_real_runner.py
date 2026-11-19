"""Real-robot runs through ProgramRunner, with a slow fake NeuraPy robot.

Run directly with Python; no Isaac Sim or NeuraPy needed.
"""

import asyncio
import importlib
import sys
import time
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch


def setUpModule():
    global ProgramRunner, ProgramBlock, NeuraPyBackend
    package_name = "_scratch_blocks_real_test"
    package = ModuleType(package_name)
    package.__path__ = [str(Path(__file__).resolve().parent)]
    imports = patch.dict(sys.modules, {
        "carb": Mock(spec=["log_info", "log_error"]),
        package_name: package,
        f"{package_name}.robot_control": Mock(),
    })
    imports.start()
    unittest.addModuleCleanup(imports.stop)
    ProgramRunner = importlib.import_module(f"{package_name}.runner").ProgramRunner
    ProgramBlock = importlib.import_module(f"{package_name}.model").ProgramBlock
    NeuraPyBackend = importlib.import_module(f"{package_name}.neurapy_backend").NeuraPyBackend


class SlowRobot:
    """Fake NeuraPy robot: every move takes 0.3 s, like a real blocking call."""
    robot_name, dof = "lara8", 6

    def __init__(self):
        self.angles = [0.0] * 6
        self.moves = []
        self.stopped = 0

    def is_robot_in_teach_mode(self):
        return False

    def set_override(self, value):
        return True

    def get_current_joint_angles(self):
        return list(self.angles)

    def move_joint(self, target_joint, **kwargs):
        time.sleep(0.3)
        self.angles = list(target_joint[-1])
        self.moves.append(kwargs.get("speed"))
        return True

    def stop(self):
        self.stopped += 1


AXES = tuple(f"lara8_joint{i}" for i in range(1, 7))


class RealRunnerTests(unittest.IsolatedAsyncioTestCase):
    def make(self, actions):
        robot = SlowRobot()
        backend = NeuraPyBackend(robot, speed_percent=10.0, acceleration_percent=10.0, override=0.2)
        blocks = [ProgramBlock(str(i), kind, value, 0, 0) for i, (kind, value) in enumerate(actions)]
        return robot, (backend, AXES), blocks

    async def test_moves_do_not_block_the_main_loop(self):
        robot, target, blocks = self.make([
            ("initialize_robot", ""),
            ("move_joint", {"joint_name": "lara8_joint1", "target_degrees": 10}),
            ("set_speed", 90.0),          # must NOT change the real speed
            ("wait", 0.1),
            ("move_joint", {"joint_name": "lara8_joint1", "target_degrees": 0}),
        ])
        finished, statuses = [], []
        runner = ProgramRunner(statuses.append, lambda _id: None,
                               on_finished=lambda ok, real: finished.append((ok, real)))
        self.assertTrue(runner.start(blocks, offline_joint_target=target))
        ticks, worst, last = 0, 0.0, time.monotonic()
        while runner.is_running:          # stand-in for Isaac Sim's 60 FPS loop
            await asyncio.sleep(1 / 60)
            now = time.monotonic()
            worst, last, ticks = max(worst, now - last), now, ticks + 1
        self.assertLess(worst, 0.1, "main loop froze during a real move")
        self.assertGreater(ticks, 20)
        self.assertEqual(robot.moves, [10.0, 10.0])   # fixed safe speed both times
        self.assertEqual(robot.stopped, 1)            # NeuraPy stop() once at the end
        self.assertEqual(finished, [(True, True)])
        self.assertTrue(any("Set Speed ignored" in s for s in statuses))

    async def test_stop_lets_current_move_finish_then_stops(self):
        robot, target, blocks = self.make([
            ("initialize_robot", ""),
            ("move_joint", {"joint_name": "lara8_joint2", "target_degrees": 5}),
            ("move_joint", {"joint_name": "lara8_joint2", "target_degrees": 10}),
            ("move_joint", {"joint_name": "lara8_joint2", "target_degrees": 15}),
        ])
        finished = []
        runner = ProgramRunner(lambda s: None, lambda _id: None,
                               on_finished=lambda ok, real: finished.append((ok, real)))
        runner.start(blocks, offline_joint_target=target)
        await asyncio.sleep(0.15)          # first move is in progress
        runner.stop()
        while runner.is_running:
            await asyncio.sleep(0.02)
        self.assertEqual(len(robot.moves), 1, "only the move in progress may complete")
        self.assertEqual(robot.stopped, 1)
        self.assertEqual(finished, [(False, True)])

    async def test_unsupported_block_fails_before_moving_further(self):
        robot, target, blocks = self.make([
            ("initialize_robot", ""),
            ("gripper", "open"),
            ("move_joint", {"joint_name": "lara8_joint1", "target_degrees": 10}),
        ])
        statuses = []
        runner = ProgramRunner(statuses.append, lambda _id: None)
        runner.start(blocks, offline_joint_target=target)
        while runner.is_running:
            await asyncio.sleep(0.02)
        self.assertEqual(robot.moves, [])
        self.assertIn("not available on the real robot", statuses[-1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
