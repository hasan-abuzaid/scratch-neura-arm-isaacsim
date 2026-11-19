import importlib.util
import json
import sys
import unittest
from pathlib import Path


MODEL_PATH = (
    Path(__file__).parents[1]
    / "exts/com.example.scratch_blocks/com/example/scratch_blocks/model.py"
)
SPEC = importlib.util.spec_from_file_location("scratch_blocks_model", MODEL_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
BlockProgram = MODULE.BlockProgram
BLOCK_STEP = MODULE.BLOCK_STEP
BLOCK_HEIGHT = MODULE.BLOCK_HEIGHT
START_HEIGHT = MODULE.START_HEIGHT


class BlockProgramTests(unittest.TestCase):
    def test_isaac_canvas_connector_pitch(self):
        self.assertEqual(BLOCK_STEP, 34)
        self.assertEqual(START_HEIGHT, 39)

    def test_snap_move_disconnect_and_round_trip(self):
        program = BlockProgram()
        first = program.add("print", 300, 200)
        second = program.add("wait", 500, 400)
        third = program.add("print", 600, 500)

        program.connect_after("START", first.id)
        program.connect_after(first.id, second.id)
        program.connect_after(second.id, third.id)
        self.assertEqual(second.y - first.y, BLOCK_STEP)
        self.assertEqual(third.y - second.y, BLOCK_STEP)
        self.assertEqual([b.id for b in program.execution_blocks()], [first.id, second.id, third.id])

        program.move_stack(second.id, 800, 300)
        self.assertEqual([b.id for b in program.execution_blocks()], [first.id])
        self.assertEqual(third.y - second.y, BLOCK_STEP)

        encoded = json.loads(json.dumps(program.to_dict()))
        restored = BlockProgram()
        restored.load_dict(encoded)
        self.assertEqual(len(restored.blocks), 3)

    def test_insert_before_middle_block(self):
        program = BlockProgram()
        first = program.add("print")
        last = program.add("print")
        middle = program.add("wait")
        program.connect_after("START", first.id)
        program.connect_after(first.id, last.id)

        program.connect_before(last.id, middle.id)

        self.assertEqual(
            [block.id for block in program.execution_blocks()],
            [first.id, middle.id, last.id],
        )
        self.assertEqual(middle.y - first.y, BLOCK_STEP)
        self.assertEqual(last.y - middle.y, BLOCK_STEP)


if __name__ == "__main__":
    unittest.main()
