"""Run directly with Python to test without Isaac Sim or NeuraPy installed."""

import math
import unittest
from unittest.mock import Mock

from neurapy_backend import NeuraPyBackend


class NeuraPyBackendTests(unittest.TestCase):
    def make_robot(self, dof):
        robot = Mock(
            spec=["robot_name", "dof", "get_current_joint_angles", "move_joint"]
        )
        robot.robot_name = f"mock-{dof}"
        robot.dof = dof
        robot.get_current_joint_angles.return_value = [0.1 * i for i in range(dof)]
        robot.move_joint.return_value = True
        return robot

    def test_single_joint_preserves_other_angles_for_both_joint_counts(self):
        for dof in (6, 7):
            with self.subTest(dof=dof):
                robot = self.make_robot(dof)
                original = robot.get_current_joint_angles.return_value[:]
                backend = NeuraPyBackend(robot)
                self.assertEqual(backend.initialize(), (f"mock-{dof}", dof))
                backend.move_single_joint(1, 15.0)
                expected = original[:]
                expected[1] = math.pi / 12
                robot.get_current_joint_angles.assert_called_once_with()
                robot.move_joint.assert_called_once_with(target_joint=[expected])
                self.assertEqual(robot.get_current_joint_angles.return_value, original)

    def test_all_joints_convert_degrees_and_keep_controller_order(self):
        robot = self.make_robot(6)
        backend = NeuraPyBackend(robot)
        backend.initialize()
        backend.move_all_joints([0, 90, -90, 180, -180, 0])
        robot.move_joint.assert_called_once_with(
            target_joint=[[0, math.pi / 2, -math.pi / 2, math.pi, -math.pi, 0]]
        )

    def test_invalid_input_never_sends_motion(self):
        robot = self.make_robot(6)
        backend = NeuraPyBackend(robot)
        with self.assertRaises(RuntimeError):
            backend.move_single_joint(1, 15.0)
        backend.initialize()
        with self.assertRaises(ValueError):
            backend.move_single_joint(6, 15.0)
        with self.assertRaises(ValueError):
            backend.move_all_joints([0] * 7)
        with self.assertRaises(ValueError):
            backend.move_all_joints([float("nan")] + [0] * 5)
        robot.get_current_joint_angles.return_value = [0] * 5
        with self.assertRaises(ValueError):
            backend.move_single_joint(1, 15.0)
        robot.move_joint.assert_not_called()

    def test_sdk_motion_failure_is_reported(self):
        robot = self.make_robot(7)
        robot.move_joint.return_value = False
        backend = NeuraPyBackend(robot)
        backend.initialize()
        with self.assertRaisesRegex(RuntimeError, "did not report"):
            backend.move_all_joints([0] * 7)


class RealModeTests(unittest.TestCase):
    """Settings used by 'Act on real': fixed slow speed, override, teach-mode refusal."""

    def make_robot(self, dof=6, teach=False):
        robot = Mock(spec=["robot_name", "dof", "get_current_joint_angles", "move_joint",
                           "is_robot_in_teach_mode", "set_override", "stop"])
        robot.robot_name, robot.dof = "lara8", dof
        robot.get_current_joint_angles.return_value = [0.0] * dof
        robot.move_joint.return_value = True
        robot.is_robot_in_teach_mode.return_value = teach
        robot.set_override.return_value = True
        return robot

    def test_speed_override_and_current_angles_are_sent(self):
        robot = self.make_robot()
        backend = NeuraPyBackend(robot, speed_percent=10.0, acceleration_percent=10.0, override=0.2)
        backend.initialize()
        robot.set_override.assert_called_once_with(0.2)
        backend.move_single_joint(0, 45.0)
        robot.move_joint.assert_called_once_with(
            target_joint=[[math.pi / 4, 0.0, 0.0, 0.0, 0.0, 0.0]],
            speed=10.0, acceleration=10.0, current_joint_angles=[0.0] * 6)

    def test_teach_mode_is_refused_before_anything_else(self):
        robot = self.make_robot(teach=True)
        backend = NeuraPyBackend(robot, speed_percent=10.0, override=0.2)
        with self.assertRaisesRegex(RuntimeError, "teach mode"):
            backend.initialize()
        robot.set_override.assert_not_called()
        with self.assertRaises(RuntimeError):
            backend.move_single_joint(0, 10.0)
        robot.move_joint.assert_not_called()

    def test_finish_calls_stop_once(self):
        robot = self.make_robot()
        NeuraPyBackend(robot).finish()
        robot.stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main(verbosity=2)
