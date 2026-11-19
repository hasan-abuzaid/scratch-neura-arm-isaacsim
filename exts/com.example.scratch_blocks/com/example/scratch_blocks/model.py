from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any
from uuid import uuid4


SUPPORTED_BLOCKS = {"set_speed", "print", "wait", "initialize_robot", "robot_properties", "move_joint", "move_all_joints", "move_linear", "move_relative", "gripper", "repeat", "end_repeat", "stop_program"}
BLOCK_WIDTH = 360
BLOCK_HEIGHT = 52
# Isaac Sim DPI-scales the SVG widget while ui.Placer uses canvas offsets.
# At Isaac Sim 6.0.1's UI scale, 34 canvas pixels is the measured socket/tab
# pitch that makes consecutive rendered blocks touch with no visible gap.
BLOCK_STEP = 34.0
START_X = 100.0
START_Y = 70.0
# Measured START-to-statement connector pitch at the same Isaac UI scale.
START_HEIGHT = 39.0
# C-shaped REPEAT: blocks between REPEAT and its END REPEAT are indented by
# this many logical pixels and drawn inside the "C".
REPEAT_INDENT = 24.0


def validate_initialize_parameters(raw_value: Any) -> dict[str, str]:
    # Version-2 programs with the old plain Prim path remain loadable.
    if isinstance(raw_value, str):
        value = {"robot_path": raw_value.strip(), "profile": "franka"}
    elif isinstance(raw_value, dict):
        value = {"robot_path": str(raw_value.get("robot_path", "")).strip(),
                 "profile": str(raw_value.get("profile", "")).strip()}
    else:
        raise ValueError("INIT ROBOT requires a Prim path and profile")
    if not value["robot_path"].startswith("/") or not value["profile"]:
        raise ValueError("INIT ROBOT requires an absolute Prim path and a profile name")
    return value


def validate_linear_parameters(raw_value: Any) -> dict[str, str | float]:
    """Validate XYZ positions/offsets in metres. The block type sets their semantics.

    New MOVE LINEAR blocks contain absolute base/world positions; MOVE RELATIVE
    contains offsets in world axes. Legacy saved linear blocks are migrated to
    MOVE RELATIVE by ProgramBlock.from_dict.
    """
    if not isinstance(raw_value, dict):
        raise ValueError("MOVE LINEAR requires a parameter dictionary")
    axis = str(raw_value.get("axis", "x")).strip().lower()
    if axis not in ("x", "y", "z"):
        raise ValueError("MOVE LINEAR axis must be x, y or z")
    distance = float(raw_value.get("distance", 0.01))
    if not math.isfinite(distance):
        raise ValueError("MOVE LINEAR distance must be finite")
    if any(key in raw_value for key in ("x", "y", "z")):
        xyz = {key: float(raw_value.get(key, 0.0)) for key in ("x", "y", "z")}
    else:
        xyz = {key: (distance if key == axis else 0.0) for key in ("x", "y", "z")}
    if not all(math.isfinite(number) for number in xyz.values()):
        raise ValueError("MOVE LINEAR XYZ values must be finite")
    coordinate_mode = str(raw_value.get("coordinate_mode", "relative")).strip()
    if coordinate_mode not in ("absolute", "relative"):
        raise ValueError("Invalid coordinate mode")
    duration = float(raw_value.get("duration", 2.0))
    raw_steps = float(raw_value.get("steps", 5))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("MOVE LINEAR duration per section must be positive")
    if not math.isfinite(raw_steps) or not raw_steps.is_integer() or raw_steps < 1:
        raise ValueError("MOVE LINEAR steps must be a positive integer")
    return {
        "x": xyz["x"], "y": xyz["y"], "z": xyz["z"],
        # Legacy keys retained for older consumers; XYZ is authoritative.
        "axis": axis, "distance": distance,
        "steps": int(raw_steps), "duration": duration,
        "robot_description_path": str(raw_value.get("robot_description_path", "")).strip(),
        "urdf_path": str(raw_value.get("urdf_path", "")).strip(),
        "end_effector_frame": str(raw_value.get("end_effector_frame", "")).strip(),
        "coordinate_mode": coordinate_mode,
    }


@dataclass
class ProgramBlock:
    id: str
    type: str
    value: str | float | dict[str, str | float]
    x: float
    y: float
    parent_id: str | None = None
    next_id: str | None = None

    @classmethod
    def create(cls, block_type: str, x: float, y: float) -> "ProgramBlock":
        if block_type == "print":
            value: str | float | dict[str, str | float] = "Hello from my visual program!"
        elif block_type == "set_speed":
            value = 10.0
        elif block_type == "wait":
            value = 1.0
        elif block_type == "repeat":
            value = 3.0
        elif block_type in ("end_repeat", "stop_program"):
            value = ""
        elif block_type == "initialize_robot":
            value = {"robot_path": "/panda", "profile": "franka"}
        elif block_type == "robot_properties":
            # Keine Eingabe erforderlich; liest den zuvor initialisierten Roboter.
            value = ""
        elif block_type == "move_linear":
            value = validate_linear_parameters({"x": 0.0, "y": 0.0, "z": 0.0, "coordinate_mode": "absolute"})
        elif block_type == "move_relative":
            value = validate_linear_parameters({"x": 0.0, "y": 0.0, "z": 0.0, "coordinate_mode": "relative"})
        elif block_type == "gripper":
            value = "open"
        elif block_type == "move_all_joints":
            value = {"targets": ""}
        elif block_type == "move_joint":
            # Gelenkname bleibt roboterspezifisch und später editierbar.
            value = {"joint_name": "Select joint", "target_degrees": 15.0}
        else:
            raise ValueError(f"Unsupported block type: {block_type}")
        return cls(str(uuid4()), block_type, value, float(x), float(y))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProgramBlock":
        block_type = str(data["type"])
        if block_type not in SUPPORTED_BLOCKS:
            raise ValueError(f"Unsupported block type: {block_type}")
        raw_value = data["value"]
        # Old MOVE LINEAR files stored relative XYZ displacements. Preserve
        # their behavior instead of silently interpreting them as coordinates.
        if block_type == "move_linear" and isinstance(raw_value, dict) and raw_value.get("coordinate_mode") != "absolute":
            block_type = "move_relative"
        if block_type == "initialize_robot":
            value = validate_initialize_parameters(raw_value)
        elif block_type == "gripper":
            value = str(raw_value).strip().lower()
            if value not in ("open", "close"):
                raise ValueError("GRIPPER action must be open or close")
        elif block_type in ("move_linear", "move_relative"):
            value = validate_linear_parameters(raw_value)
            value["coordinate_mode"] = "absolute" if block_type == "move_linear" else "relative"
        elif block_type == "move_all_joints":
            if not isinstance(raw_value, dict):
                raise ValueError("MOVE ALL requires targets")
            value = {"targets": str(raw_value.get("targets", ""))}
            parse_joint_targets(value["targets"])
        elif block_type == "move_joint":
            if not isinstance(raw_value, dict):
                raise ValueError("MOVE JOINT requires a parameter dictionary")
            joint_name = raw_value.get("joint_name")
            if not isinstance(joint_name, str) or not joint_name.strip():
                raise ValueError("MOVE JOINT requires a joint name")
            target_degrees = float(raw_value["target_degrees"])
            if not math.isfinite(target_degrees):
                raise ValueError("MOVE JOINT target must be finite")
            value: str | float | dict[str, str | float] = {
                "joint_name": joint_name.strip(),
                "target_degrees": target_degrees,
            }
        elif block_type == "set_speed":
            value = validate_speed(raw_value)
        elif block_type == "wait":
            value = max(0.0, float(raw_value))
        elif block_type == "repeat":
            count = float(raw_value)
            if not math.isfinite(count) or not count.is_integer() or not (1 <= count <= 100):
                raise ValueError("REPEAT count must be a whole number from 1 to 100")
            value = count
        elif block_type in ("end_repeat", "stop_program"):
            value = ""
        else:
            value = str(raw_value)
        return cls(
            id=str(data.get("id") or uuid4()),
            type=block_type,
            value=value,
            x=float(data.get("x", START_X)),
            y=float(data.get("y", START_Y + START_HEIGHT)),
            parent_id=data.get("parent_id"),
            next_id=data.get("next_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BlockProgram:
    """Coordinate-based blocks with one Scratch-style stack attached to START."""

    def __init__(self) -> None:
        self.blocks: dict[str, ProgramBlock] = {}
        self.start_next: str | None = None

    def add(self, block_type: str, x: float = 200, y: float = 180) -> ProgramBlock:
        block = ProgramBlock.create(block_type, x, y)
        self.blocks[block.id] = block
        return block

    def get(self, block_id: str) -> ProgramBlock:
        try:
            return self.blocks[block_id]
        except KeyError as error:
            raise ValueError(f"Unknown block id: {block_id}") from error

    def chain_ids(self, first_id: str) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        current: str | None = first_id
        while current is not None:
            if current in seen:
                raise ValueError("Cycle detected in block program")
            seen.add(current)
            result.append(current)
            current = self.get(current).next_id
        return result

    def execution_blocks(self) -> list[ProgramBlock]:
        if self.start_next is None:
            return []
        return [self.get(block_id) for block_id in self.chain_ids(self.start_next)]

    def detach(self, block_id: str) -> None:
        block = self.get(block_id)
        if block.parent_id == "START":
            self.start_next = None
        elif block.parent_id:
            self.get(block.parent_id).next_id = None
        block.parent_id = None

    def move_stack(self, block_id: str, x: float, y: float) -> None:
        self.detach(block_id)
        ids = self.chain_ids(block_id)
        for index, current_id in enumerate(ids):
            block = self.get(current_id)
            block.x = float(x)
            block.y = float(y) + index * BLOCK_STEP

    def connect_after(self, target_id: str, block_id: str) -> None:
        moving_ids = self.chain_ids(block_id)
        if target_id != "START" and target_id in moving_ids:
            return

        self.detach(block_id)
        if target_id == "START":
            displaced_id = self.start_next
            self.start_next = block_id
            first_x = START_X
            first_y = START_Y + START_HEIGHT
        else:
            target = self.get(target_id)
            displaced_id = target.next_id
            target.next_id = block_id
            first_x = target.x
            first_y = target.y + BLOCK_STEP

        first = self.get(block_id)
        first.parent_id = target_id

        tail = self.get(moving_ids[-1])
        if displaced_id is not None and displaced_id not in moving_ids:
            tail.next_id = displaced_id
            self.get(displaced_id).parent_id = tail.id

        full_ids = self.chain_ids(block_id)
        for index, current_id in enumerate(full_ids):
            current = self.get(current_id)
            current.x = first_x
            current.y = first_y + index * BLOCK_STEP

    def connect_before(self, target_id: str, block_id: str) -> None:
        """Insert a block/stack directly before target, preserving the chain."""
        moving_ids = self.chain_ids(block_id)
        if target_id in moving_ids:
            return

        target = self.get(target_id)
        parent_id = target.parent_id
        target_x, target_y = target.x, target.y
        self.detach(block_id)

        first = self.get(block_id)
        first.parent_id = parent_id
        if parent_id == "START":
            self.start_next = block_id
        elif parent_id:
            self.get(parent_id).next_id = block_id

        tail = self.get(moving_ids[-1])
        tail.next_id = target_id
        target.parent_id = tail.id

        for index, current_id in enumerate(self.chain_ids(block_id)):
            current = self.get(current_id)
            current.x = target_x
            current.y = target_y + index * BLOCK_STEP

    def remove(self, block_id: str) -> None:
        block = self.get(block_id)
        successor_id = block.next_id
        parent_id = block.parent_id
        if parent_id == "START":
            self.start_next = successor_id
        elif parent_id:
            self.get(parent_id).next_id = successor_id
        if successor_id:
            self.get(successor_id).parent_id = parent_id
        del self.blocks[block_id]
        if successor_id and parent_id:
            self._reflow_from(successor_id)

    def _reflow_from(self, block_id: str) -> None:
        block = self.get(block_id)
        if block.parent_id == "START":
            x, y = START_X, START_Y + START_HEIGHT
        elif block.parent_id:
            parent = self.get(block.parent_id)
            x, y = parent.x, parent.y + BLOCK_STEP
        else:
            return
        for index, current_id in enumerate(self.chain_ids(block_id)):
            current = self.get(current_id)
            current.x, current.y = x, y + index * BLOCK_STEP

    # ------------------------------------------------------------------
    # C-shaped REPEAT support. The stored program stays a flat chain with
    # REPEAT ... END REPEAT pairs (runner and save format unchanged); these
    # helpers only control layout and which blocks move together.
    # ------------------------------------------------------------------
    def matching_end(self, repeat_id: str) -> ProgramBlock | None:
        """END REPEAT that closes the given REPEAT, or None if unmatched."""
        repeat = self.get(repeat_id)
        if repeat.type != "repeat" or repeat.next_id is None:
            return None
        depth = 0
        for block_id in self.chain_ids(repeat.next_id):
            block = self.get(block_id)
            if block.type == "repeat":
                depth += 1
            elif block.type == "end_repeat":
                if depth == 0:
                    return block
                depth -= 1
        return None

    def segment_ids(self, first_id: str) -> list[str]:
        """Blocks that move with first_id: everything below it, but never the
        END REPEAT of a C that first_id sits inside (that arm stays behind)."""
        ids: list[str] = []
        depth = 0
        for block_id in self.chain_ids(first_id):
            block = self.get(block_id)
            if block.type == "end_repeat":
                if depth == 0 and block_id != first_id:
                    break
                depth = max(0, depth - 1)
            ids.append(block_id)
            if block.type == "repeat":
                depth += 1
        return ids

    def detach_segment(self, first_id: str) -> list[str]:
        """Cut segment_ids(first_id) out of its chain and close the gap."""
        ids = self.segment_ids(first_id)
        first, last = self.get(ids[0]), self.get(ids[-1])
        after_id, parent_id = last.next_id, first.parent_id
        if parent_id == "START":
            self.start_next = after_id
        elif parent_id:
            self.get(parent_id).next_id = after_id
        if after_id is not None:
            self.get(after_id).parent_id = parent_id
        last.next_id = None
        first.parent_id = None
        return ids

    def move_segment(self, first_id: str, x: float, y: float) -> None:
        """Detach a segment and place its first block at (x, y)."""
        self.detach_segment(first_id)
        block = self.get(first_id)
        block.x, block.y = float(x), float(y)
        self.layout()

    def layout(self) -> None:
        """Position every stack: one BLOCK_STEP per block, indented inside REPEATs.

        An empty REPEAT keeps one free slot so blocks can be dropped into it.
        Stack roots keep their position; only the blocks below them move.
        """
        roots: list[tuple[str, float, float]] = []
        if self.start_next is not None:
            roots.append((self.start_next, START_X, START_Y + START_HEIGHT))
        for block in self.blocks.values():
            if block.parent_id is None:
                roots.append((block.id, block.x, block.y))
        for first_id, x0, y0 in roots:
            depth, y = 0, y0
            for block_id in self.chain_ids(first_id):
                block = self.get(block_id)
                if block.type == "end_repeat" and depth > 0:
                    depth -= 1
                block.x, block.y = x0 + depth * REPEAT_INDENT, y
                y += BLOCK_STEP
                if block.type == "repeat":
                    depth += 1
                    following = self.blocks.get(block.next_id) if block.next_id else None
                    if following is not None and following.type == "end_repeat":
                        y += BLOCK_STEP  # empty C: leave a visible mouth

    def clear(self) -> None:
        self.blocks.clear()
        self.start_next = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "start_next": self.start_next,
            "blocks": [block.to_dict() for block in self.blocks.values()],
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        if int(data.get("version", 0)) != 2:
            raise ValueError("This canvas requires a version 2 program")
        blocks = [ProgramBlock.from_dict(item) for item in data.get("blocks", [])]
        self.blocks = {block.id: block for block in blocks}
        self.start_next = data.get("start_next")
        if self.start_next is not None:
            self.execution_blocks()  # validate references and cycles


def parse_joint_targets(text):
    """Parse explicit name=angle pairs in degrees, separated by commas."""
    result = {}
    for pair in str(text).split(","):
        name, separator, angle = pair.strip().partition("=")
        name = name.strip()
        if not separator or not name or name in result:
            raise ValueError("Enter each joint once: name=degrees, name=degrees")
        target = float(angle.strip())
        if not math.isfinite(target):
            raise ValueError("Joint targets must be finite")
        result[name] = target
    return result


def _joint_number(name: str) -> int | None:
    """Trailing number of a joint name: 'lara8_joint2' -> 2, 'shoulder' -> None."""
    import re
    match = re.search(r"(\d+)\s*$", str(name))
    return int(match.group(1)) if match else None


def retarget_joint_names(program: "BlockProgram", new_names: list[str]) -> tuple[int, list[str]]:
    """Point joint blocks at another robot's joints, matched by joint NUMBER.

    A program written for lara8 ('lara8_joint2') opened on a MAiRA becomes
    'maira7M_joint2'. Names already valid stay. Returns (changed, missing):
    missing = joints with no same-numbered joint on the new robot (e.g. joint7
    on a 6-joint arm); those blocks are left unchanged and will fail Compile.
    """
    by_number: dict[int, str] = {}
    duplicates: set[int] = set()
    for name in new_names:
        number = _joint_number(name)
        if number is None:
            continue
        if number in by_number:
            duplicates.add(number)
        by_number[number] = name
    for number in duplicates:
        by_number.pop(number, None)   # ambiguous numbering: never guess

    def mapped(name: str) -> str | None:
        if name in new_names:
            return name
        number = _joint_number(name)
        return by_number.get(number) if number is not None else None

    changed, missing = 0, set()
    for block in program.blocks.values():
        if block.type == "move_joint" and isinstance(block.value, dict):
            old = str(block.value.get("joint_name", ""))
            if not old or old == "Select joint":
                continue
            new = mapped(old)
            if new is None:
                missing.add(old)
            elif new != old:
                block.value = {**block.value, "joint_name": new}
                changed += 1
        elif block.type == "move_all_joints" and isinstance(block.value, dict):
            try:
                targets = parse_joint_targets(block.value.get("targets", ""))
            except ValueError:
                continue
            new_targets, block_changed = {}, False
            for old, degrees in targets.items():
                new = mapped(old)
                if new is None:
                    missing.add(old)
                    new = old
                block_changed |= new != old
                new_targets[new] = degrees
            if block_changed:
                block.value = {"targets": ", ".join(f"{n}={d:g}" for n, d in new_targets.items())}
                changed += 1
    return changed, sorted(missing)


def validate_speed(value):
    """Maximum planned rotational joint target speed, in degrees/second."""
    speed = float(value)
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("Speed must be finite and greater than zero (deg/s)")
    return speed
