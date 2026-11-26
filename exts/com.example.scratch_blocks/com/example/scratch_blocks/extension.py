from __future__ import annotations

import asyncio
import json
import math
import os
import time
from typing import Any, Callable

import carb.tokens
import carb.input
import omni.ext
import omni.ui as ui

from .model import (
    BLOCK_STEP,
    BLOCK_HEIGHT,
    BLOCK_WIDTH,
    REPEAT_INDENT,
    START_HEIGHT,
    START_X,
    START_Y,
    BlockProgram,
    ProgramBlock,
    SUPPORTED_BLOCKS,
    validate_speed,
    parse_joint_targets,
    validate_linear_parameters,
    validate_initialize_parameters,
    validate_wave_parameters,
)
from .runner import ProgramRunner


WINDOW_TITLE = "Scratch Robot Blocks"
DRAG_PREFIX = "scratch-canvas:"
CANVAS_WIDTH = 2400   # whiteboard size (logical px); scroll to reach the rest
CANVAS_HEIGHT = 1600
# Snap zone (logical canvas pixels), measured from the connector under a block:
# a dropped block connects when its top edge lands from SNAP_ABOVE above the
# connector down to SNAP_BELOW below it, i.e. "directly under" that block.
SNAP_ABOVE = 20.0
SNAP_BELOW = 50.0
SNAP_X_TOLERANCE = 120.0
TEMPLATE_WIDTH = 260.0   # size of the blocks in the library on the left
TEMPLATE_HEIGHT = 38.0
GRID_SIZE = 100
# ---- Look & feel (omni.ui colours are 0xAABBGGRR) -------------------------
PANEL_BG = 0xFF2A2724        # dark panels, matches Isaac Sim
PANEL_BORDER = 0xFF3D3935
TEXT_MUTED = 0xFF9A948E
WHITEBOARD = 0xFFFFFFFF
GRID_MINOR = 0xFFF1EEEC
GRID_MAJOR = 0xFFE0DBD7      # every 5th line, like squared paper
BTN = 0xFF46413C
BTN_HOVER = 0xFF5A544E
BTN_RUN = 0xFF43A02E         # green
BTN_STOP = 0xFF3336DA        # red
BTN_REAL = 0xFF227EE6        # orange: anything that touches the real robot
CATEGORY_COLORS = {"ROBOT": 0xFF2F7CE8, "MOTION": 0xFFF35A96, "GRIPPER": 0xFFC255C0,
                   "CONTROL": 0xFF2E9AD9}
ZOOM_MIN = 0.50
ZOOM_MAX = 2.00
ZOOM_STEP = 0.25
# Vertical placement of fields and the X button inside a block. The block
# image inherits the window margin, so its coloured body is drawn from about
# y+14 to y+44 (not 0..44 as in the SVG). These values keep widgets inside it.
WIDGET_TOP = 17.0
WIDGET_HEIGHT = 24.0
# Step 2 (live drag): an invisible draggable handle covers the block's title
# area (coloured body runs from about y+14 to y+44). Fields are never covered.
HANDLE_TOP = 14.0
HANDLE_HEIGHT = 30.0
# The grab area itself is a bit larger than the coloured body and has margin 0.
# Before, it inherited the window margin of 7 px on every side, so only a
# ~16 px band in the middle of the title caught the mouse ("grab works on the
# 3rd try"). Set SHOW_GRAB_AREAS = True to see the grab areas in magenta.
GRAB_TOP = 10.0
GRAB_HEIGHT = 38.0
GRAB_EXTRA_WIDTH = 14.0
SHOW_GRAB_AREAS = False
# Saved projects live here, so they survive sessions and can be copied to other PCs.
PROJECTS_DIR = os.path.join(os.path.expanduser("~"), "Documents", "ScratchRobotProjects")  # default
# The folder you choose is remembered across Isaac Sim restarts (carb persistent setting).
PROJECTS_DIR_SETTING = "/persistent/exts/com.example.scratch_blocks/projects_dir"
# Mouse travel (logical px) below which a press counts as a click, not a move.
DRAG_CLICK_TOLERANCE = 3.0
# Each block now sits inside one extra Placer + ZStack. Both inherit the
# window style {"margin": 7}, which shifted every block 2 x 7 = 14 screen px
# right and down (measured: START left edge 175 px, first block 190 px).
# Margins are not zoom-scaled, so this compensation is constant screen pixels.
WRAPPER_MARGIN_PX = 14.0
# C-shaped REPEAT: colour of the side bar (omni.ui colours are 0xAABBGGRR;
# this is RGB 197,139,50 sampled from repeat_block.svg) and arm width.
REPEAT_COLOR = 0xFF328BC5
END_ARM_WIDTH = 200.0
# WAVE block: joint menu + four number fields need a wider block than BLOCK_WIDTH.
WAVE_WIDTH = 520.0
WAVE_FIELDS = (("low", "low", "Lower end of the swing (degrees)"),
               ("high", "high", "Upper end of the swing (degrees); the wave starts here"),
               ("frequency", "Hz", "Full swings per second (simulation; the real arm uses its fixed speed)"),
               ("cycles", "x", "Number of full swings: 0.5, 1, 1.5 ... (whole = ends at high)"))
# Box selection: outline colour of selected blocks and the selection box
# (0xAABBGGRR: RGB 30,158,255 blue; box fill is the same blue, mostly transparent).
SELECT_COLOR = 0xFFFF9E1E
SELECT_FILL = 0x33FF9E1E
# Mouse travel (screen px) below which a press on empty canvas is a click.
MARQUEE_CLICK_TOLERANCE = 4.0


class _LocalBlock:
    """View of a ProgramBlock positioned at (0, 0).

    Each block is now drawn inside its own Placer, so its layers use local
    coordinates. Everything except x/y is forwarded to the real block, and
    writes (e.g. block.value = ...) go to the real block.
    """

    __slots__ = ("_block",)

    def __init__(self, block: Any) -> None:
        object.__setattr__(self, "_block", block)

    def __getattr__(self, name: str) -> Any:
        if name in ("x", "y"):
            return 0.0
        return getattr(self._block, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._block, name, value)


class ScratchBlocksExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        self._window = ScratchBlocksWindow(WINDOW_TITLE, width=1150, height=720)

    def on_shutdown(self) -> None:
        if self._window:
            self._window.destroy()
            self._window = None


class ScratchBlocksWindow(ui.Window):
    def __init__(self, title: str, **kwargs: Any) -> None:
        super().__init__(title, **kwargs)
        self._program = BlockProgram()
        self._active_block_id: str | None = None
        self._delete_pending = False
        self._parameter_window = None
        self._status_label: ui.Label | None = None
        self._canvas_frame: ui.Frame | None = None
        self._canvas_surface: ui.Rectangle | None = None
        self._zoom_label: ui.Label | None = None
        self._zoom = 1.0
        self._drag_anchor_x = 0.0
        self._drag_anchor_y = 0.0
        self._templates: dict[str, ui.ZStack] = {}
        self._drop_handled = False
        self._press: tuple[float, float, float, float] | None = None
        self._runner = ProgramRunner(self._set_status, self._set_active_block,
                                     on_finished=self._on_run_finished,
                                     on_real_gripper=self._mirror_real_gripper)

        # --- Act on real robot -------------------------------------------
        self._sim_arm = None            # Isaac Sim handle created by Sync
        self._real_reader = None        # NeuraPy connection used for reading/mirror
        self._mirror = None             # RealArmMirror while acting on the real arm
        self._sync_pose: list[float] | None = None   # real pose copied into the sim
        self._pending_validation: dict | None = None # set when a sim run starts at the synced pose
        self._validation: dict | None = None         # {fingerprint, start} after a successful sim run
        self._act_confirm_until = 0.0
        self._real_busy = False
        self._act_button = None
        self._asset_dir = os.path.join(os.path.dirname(__file__), "assets")

        data_root = carb.tokens.get_tokens_interface().resolve("${data}")
        self._save_path = os.path.join(data_root, "scratch_blocks_demo", "canvas_program.json")
        self._robot_menu = None
        self._gripper_init_menu = None
        self._joint_menu = None
        self._gripper_menu = None
        self._open_menu_widget = None
        self._project_name_model = ui.SimpleStringModel("my_program")
        self._projects_dir = self._load_projects_dir()
        self._folder_window = None
        # Dropdown labels on blocks, updated in place instead of rebuilding.
        self._joint_labels: dict[str, Any] = {}
        self._gripper_labels: dict[str, Any] = {}
        self._joint_degree_models = {}
        self._joint_degree_fields = {}
        self._linear_models = {}
        self._linear_fields = {}
        self._all_joint_models = {}
        self._all_joint_fields = {}
        self._speed_models = {}
        self._speed_fields = {}
        self._repeat_models = {}
        self._repeat_fields = {}
        self._wave_models = {}   # (block_id, key) -> model of a WAVE number field
        self._wave_fields = {}
        # Step 1 (performance): highlight frames are built once per canvas
        # build and only shown/hidden while a program runs.
        self._highlight_rects: dict[str, Any] = {}
        # Step 1 (performance): stage scans and profile files are cached
        # instead of being re-read on every canvas rebuild.
        self._joint_cache: dict[str, list[str]] = {}
        self._profile_cache: dict[str, dict] = {}
        # Step 2 (live drag): one visual Placer and one drag handle per block.
        self._block_placers: dict[str, Any] = {}
        self._handle_placers: dict[str, Any] = {}
        self._drag: dict[str, Any] | None = None
        # Box selection: selected block ids survive canvas rebuilds.
        self._selected: set[str] = set()
        self._select_rects: dict[str, Any] = {}
        self._marquee_placer = None
        self._marquee_rect = None
        self._marquee_stack = None
        self._marquee_active = False
        self._marquee_cancelled = False
        self._canvas_stack = None
        self._block_images: dict[str, Any] = {}
        self._mouse_scale: tuple[float, float] | None = None
        self.frame.set_build_fn(self._build_ui)

    def destroy(self) -> None:
        self._gripper_init_menu = None
        self._robot_menu = None
        self._joint_menu = None
        self._gripper_menu = None
        self._joint_degree_models.clear()
        self._joint_degree_fields.clear()
        self._linear_models.clear()
        self._linear_fields.clear()
        self._all_joint_models.clear()
        self._all_joint_fields.clear()
        self._speed_models.clear()
        self._speed_fields.clear()
        self._repeat_models.clear()
        self._repeat_fields.clear()
        self._wave_models.clear()
        self._wave_fields.clear()
        self._highlight_rects.clear()
        self._invalidate_scene_cache()
        self._runner.stop()
        if self._mirror is not None:
            self._mirror.stop()
            self._mirror = None
        if self._parameter_window is not None:
            self._parameter_window.visible = False
            self._parameter_window = None
        super().destroy()

    def _build_ui(self) -> None:
        with ui.VStack(spacing=6, style={"margin": 7}):
            self._build_toolbar()
            with ui.HStack(spacing=8):
                self._build_library()
                with ui.VStack(spacing=4):
                    with ui.HStack(height=22):
                        ui.Label("WORKSPACE", width=100, style={"font_size": 15})
                        ui.Label("drag blocks by their title  ·  drag on empty space to select  ·  "
                                 "Ctrl + wheel to zoom", style={"color": TEXT_MUTED, "font_size": 13})
                    with ui.ZStack():
                        ui.Rectangle(style={"background_color": PANEL_BORDER, "border_radius": 6})
                        with ui.ScrollingFrame(
                            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                            style={"background_color": WHITEBOARD},
                        ):
                            self._canvas_frame = ui.Frame()
                            self._canvas_frame.set_build_fn(self._build_canvas)
            self._build_status_bar()

    @staticmethod
    def _button_style(color: int) -> dict:
        hover = BTN_HOVER if color == BTN else (color & 0x00FFFFFF) | 0xCC000000
        return {"Button": {"background_color": color, "border_radius": 5, "margin": 2},
                "Button:hovered": {"background_color": hover},
                "Button.Label": {"color": 0xFFFFFFFF}}

    def _tool_button(self, text: str, clicked_fn, width: int = 70, color: int = BTN,
                     tooltip: str = "") -> Any:
        return ui.Button(text, width=width, height=28, clicked_fn=clicked_fn, tooltip=tooltip,
                         style=self._button_style(color))

    @staticmethod
    def _group_label(text: str) -> None:
        ui.Label(text, width=0, style={"color": TEXT_MUTED, "font_size": 12, "margin_width": 4})

    @staticmethod
    def _group_divider() -> None:
        ui.Spacer(width=6)
        ui.Rectangle(width=1, style={"background_color": PANEL_BORDER})
        ui.Spacer(width=6)

    def _build_toolbar(self) -> None:
        with ui.ZStack(height=42):
            ui.Rectangle(style={"background_color": PANEL_BG, "border_radius": 6})
            with ui.HStack(spacing=4, style={"margin": 5}):
                self._group_label("PROGRAM")
                self._tool_button("Compile", self._compile, 72,
                                  tooltip="Check the program without running it")
                self._tool_button("Run", self._run, 64, BTN_RUN,
                                  tooltip="Run the program on the simulated robot")
                self._tool_button("Stop", self._runner.stop, 60, BTN_STOP,
                                  tooltip="Stop after the current block")
                self._group_divider()
                self._group_label("REAL ROBOT")
                self._tool_button("Sync", self._sync_from_real, 60, BTN_REAL,
                                  tooltip="Read the real arm and put the simulated arm in the same pose")
                self._act_button = self._tool_button(
                    "Act on real (locked)", self._act_on_real, 150, BTN_REAL,
                    tooltip="Runs the program on the REAL robot. Unlocks only after this exact\n"
                            "program completed in simulation from the synced pose.")
                self._group_divider()
                self._group_label("PROJECT")
                ui.StringField(self._project_name_model, width=140, height=26,
                               tooltip="Name to save this program under",
                               style={"border_radius": 4, "margin": 2})
                self._tool_button("Save", self._save, 56,
                                  tooltip="Save to the projects folder (change it under Open)")
                self._tool_button("Open", self._open_menu, 56, tooltip="Open a saved project")
                self._tool_button("Clear", self._clear, 56, tooltip="Remove all blocks")
                ui.Spacer()
                self._group_label("ZOOM")
                self._tool_button("−", lambda: self._change_zoom(-ZOOM_STEP), 30)
                self._zoom_label = ui.Label("100%", width=48, alignment=ui.Alignment.CENTER)
                self._tool_button("+", lambda: self._change_zoom(ZOOM_STEP), 30)
                self._tool_button("Fit", lambda: self._set_zoom(1.0), 40, tooltip="Reset zoom to 100 %")

    def _build_status_bar(self) -> None:
        with ui.ZStack(height=26):
            ui.Rectangle(style={"background_color": PANEL_BG, "border_radius": 5})
            with ui.HStack(style={"margin": 4}):
                ui.Label("STATUS", width=56, style={"color": TEXT_MUTED, "font_size": 12})
                self._status_label = ui.Label("Ready", style={"font_size": 13})

    LIBRARY = (
        ("ROBOT", (("initialize_robot", "Initialize Robot", "initialize_robot_block.svg"),)),
        ("MOTION", (("move_joint", "Move Single Joint", "move_joint_block.svg"),
                    ("move_all_joints", "Move All Joints", "move_joint_block.svg"),
                    ("move_linear", "Move Linear (absolute)", "move_linear_block.svg"),
                    ("move_relative", "Move Relative", "move_linear_block.svg"),
                    ("wave", "Wave (preset)", "move_joint_block.svg"),
                    ("set_speed", "Set Joint Speed (deg/s)", "move_joint_block.svg"))),
        ("GRIPPER", (("initialize_gripper", "Initialize Gripper", "initialize_robot_block.svg"),
                     ("gripper", "Gripper open / close", "gripper_block.svg"))),
        ("CONTROL", (("wait", "Wait seconds", "wait_block.svg"),
                     ("repeat", "Repeat (box)", "repeat_block.svg"),
                     ("stop_program", "Stop Program", "stop_program_block.svg"))),
    )

    def _build_library(self) -> None:
        with ui.ZStack(width=TEMPLATE_WIDTH + 40):
            ui.Rectangle(style={"background_color": PANEL_BG, "border_radius": 6})
            with ui.VStack(spacing=4, style={"margin": 6}):
                ui.Label("BLOCKS", height=22, style={"font_size": 15})
                ui.Label("drag onto the workspace", height=16,
                         style={"color": TEXT_MUTED, "font_size": 12})
                with ui.ScrollingFrame(
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    style={"background_color": PANEL_BG},
                ):
                    with ui.VStack(spacing=6, height=0):
                        for category, items in self.LIBRARY:
                            with ui.HStack(height=24, spacing=6):
                                ui.Rectangle(width=4, style={"background_color": CATEGORY_COLORS[category],
                                                             "border_radius": 2})
                                ui.Label(category, style={"color": TEXT_MUTED, "font_size": 12})
                            for block_type, label, asset in items:
                                self._build_template(block_type, label, asset)
                            ui.Spacer(height=4)

    def _build_template(self, block_type: str, label: str, asset: str) -> None:
        payload = lambda t=block_type: self._template_payload(t)
        template = ui.ZStack(
            width=ui.Pixel(TEMPLATE_WIDTH),
            height=ui.Pixel(TEMPLATE_HEIGHT),
            drag_fn=payload,
            mouse_pressed_fn=lambda x, y, button, modifier, t=block_type: self._remember_template_anchor(x, y, t),
        )
        self._templates[block_type] = template
        with template:
            ui.Image(
                os.path.join(self._asset_dir, asset),
                width=ui.Pixel(260),
                height=ui.Pixel(38),
                fill_policy=ui.FillPolicy.STRETCH,
                drag_fn=payload,
            )
            ui.Label(
                label,
                alignment=ui.Alignment.CENTER,
                drag_fn=payload,
                style={"color": 0xFFFFFFFF, "font_size": 13},
            )

    def _build_canvas(self) -> None:
        self._joint_degree_fields.clear()
        self._all_joint_fields.clear()
        self._speed_fields.clear()
        self._repeat_fields.clear()
        self._wave_fields.clear()
        self._highlight_rects.clear()
        self._joint_labels.clear()
        self._gripper_labels.clear()
        self._block_placers.clear()
        self._handle_placers.clear()
        build_started = time.perf_counter()
        self._drag = None
        self._select_rects.clear()
        self._selected &= set(self._program.blocks)  # forget deleted blocks
        # Indent blocks inside REPEAT boxes and stack every chain tidily.
        self._program.layout()
        scale = self._zoom
        self._block_images.clear()
        self._canvas_stack = ui.ZStack(
            width=ui.Pixel(CANVAS_WIDTH * scale),
            height=ui.Pixel(CANVAS_HEIGHT * scale),
        )
        with self._canvas_stack:
            self._canvas_surface = ui.Rectangle(
                style={"background_color": WHITEBOARD},
                accept_drop_fn=self._accept_drop,
                drop_fn=self._drop_on_canvas,
                mouse_wheel_fn=self._on_canvas_wheel,
                mouse_pressed_fn=self._canvas_pressed,
            )
            self._build_grid()
            self._build_start()
            # Step 2: every block is one movable Placer holding all its layers.
            # Lower blocks are drawn first, so a block's fields stay above the
            # transparent top margin of the block connected under it.
            for block in sorted(self._program.blocks.values(), key=lambda b: -b.y):
                self._build_block_layers(block)
            # Drag handles are topmost so nothing can intercept them.
            for block in self._program.blocks.values():
                if block.type != "end_repeat":  # the C's arm moves with its REPEAT
                    self._build_drag_handle(block)
            self._build_marquee()
        elapsed_ms = (time.perf_counter() - build_started) * 1000.0
        print(f"[Scratch Blocks][perf] canvas rebuild {elapsed_ms:.1f} ms "
              f"for {len(self._program.blocks)} block(s)")

    def _build_block_layers(self, block: ProgramBlock) -> None:
        """Draw one block (image, fields, X button, highlight) in its own Placer."""
        scale = self._zoom
        width = self._width_for_block(block)
        local = _LocalBlock(block)
        span = 0.0
        if block.type == "repeat":
            end = self._program.matching_end(block.id)
            span = (end.y - block.y) if end is not None else 0.0
        wrap_w = ui.Pixel((width + 6) * scale)
        wrap_h = ui.Pixel((BLOCK_HEIGHT + 6 + span) * scale)
        # Explicit size on the Placer too: without it the container stretches
        # over the canvas and its invisible area steals mouse presses.
        placer = ui.Placer(
            offset_x=ui.Pixel(self._visual_offset(block.x)),
            offset_y=ui.Pixel(self._visual_offset(block.y)),
            width=wrap_w, height=wrap_h,
            stable_size=True,
        )
        self._block_placers[block.id] = placer
        with placer:
            with ui.ZStack(width=wrap_w, height=wrap_h):
                if span > 0:
                    self._build_repeat_spine(span)  # drawn first: header and arm cover its ends
                self._build_block(local)
                self._build_selection_outline(block, width)
                self._build_editable_field(local)
                if block.type != "end_repeat":
                    self._build_delete_button(local)
                if block.type == "move_joint":
                    self._build_joint_degree_field(local)
                elif block.type in ("move_linear", "move_relative"):
                    self._build_linear_fields(local)
                    self._build_linear_advanced_button(local)
                elif block.type == "move_all_joints":
                    self._build_all_joint_fields(local)
                elif block.type == "set_speed":
                    self._build_speed_field(local)
                elif block.type == "repeat":
                    self._build_repeat_field(local)
                elif block.type == "wave":
                    self._build_wave_fields(local)

    # ------------------------------------------------------------------
    # Step 2: live dragging. The handle Placer follows the mouse by itself
    # (draggable=True); its offset callbacks move the block and every block
    # connected below it. Releasing the mouse snaps with _nearest_insertion.
    # ------------------------------------------------------------------
    def _build_repeat_spine(self, span: float) -> None:
        """Left bar of the C, from the REPEAT header down to its END arm.

        It uses the same Placer -> ZStack -> widget nesting as the block image,
        so it inherits the same margins and lines up with the block bodies.
        The bar starts and ends inside the header/arm bodies (y 14..44), which
        are drawn on top of it, so small vertical offsets stay invisible.
        """
        scale = self._zoom
        top, bottom = 30.0, span + 30.0
        # Every level gets an explicit pixel size: a Rectangle without one
        # stretches to fill the whole canvas (seen in Isaac Sim 6.2).
        width = ui.Pixel(REPEAT_INDENT * scale + 14)
        height = ui.Pixel((bottom - top) * scale + 14)
        with ui.Placer(offset_x=ui.Pixel(0), offset_y=ui.Pixel(top * scale),
                       width=width, height=height, stable_size=True):
            with ui.ZStack(width=width, height=height):
                ui.Rectangle(width=width, height=height,
                             opaque_for_mouse_events=False,
                             style={"background_color": REPEAT_COLOR, "border_radius": 0})

    def _visual_offset(self, logical: float) -> float:
        """Screen offset for a block wrapper at a logical canvas coordinate."""
        return logical * self._zoom - WRAPPER_MARGIN_PX

    def _label_width(self, block: ProgramBlock) -> float:
        """Logical width of the block's title label (the draggable area)."""
        if block.type == "move_all_joints":
            return 110.0
        if block.type == "robot_properties":
            return 178.0
        if block.type == "stop_program":
            return 126.0
        if block.type == "initialize_gripper":
            return 100.0
        if block.type == "wave":
            return 50.0
        if block.type in ("initialize_robot", "set_speed", "move_joint", "move_linear",
                          "move_relative", "gripper", "repeat", "end_repeat"):
            return 90.0
        return 43.0

    def _build_drag_handle(self, block: ProgramBlock) -> None:
        scale = self._zoom
        width = (self._label_width(block) + GRAB_EXTRA_WIDTH) * scale
        handle_w, handle_h = ui.Pixel(width), ui.Pixel(GRAB_HEIGHT * scale)
        # Explicit size at every level. Without it the transparent handle
        # stretched over the blocks below, so grabbing SINGLE JOINT under a
        # REPEAT actually grabbed the REPEAT (and dragged its whole box).
        placer = ui.Placer(
            offset_x=ui.Pixel(block.x * scale),
            offset_y=ui.Pixel((block.y + GRAB_TOP) * scale),
            width=handle_w, height=handle_h,
            draggable=not self._runner.is_running,
            stable_size=True,
            style={"margin": 0},   # the WHOLE area catches the mouse, not a shrunken band
        )
        self._handle_placers[block.id] = placer
        with placer:
            ui.Rectangle(
                width=handle_w,
                height=handle_h,
                tooltip="Drag to move this block and the blocks below it",
                accept_drop_fn=self._accept_drop,
                drop_fn=self._drop_on_canvas,
                mouse_wheel_fn=self._on_canvas_wheel,
                style={"margin": 0,
                       "background_color": 0x55FF00FF if SHOW_GRAB_AREAS else 0x00000000},
            )
        placer.set_offset_x_changed_fn(lambda _value, b_id=block.id: self._handle_moved(b_id))
        placer.set_offset_y_changed_fn(lambda _value, b_id=block.id: self._handle_moved(b_id))

    def _handle_offset(self, block_id: str) -> tuple[float, float] | None:
        """Current handle position as the block's logical top-left corner."""
        placer = self._handle_placers.get(block_id)
        if placer is None:
            return None
        scale = self._zoom
        return (float(placer.offset_x.value) / scale,
                float(placer.offset_y.value) / scale - GRAB_TOP)

    def _handle_moved(self, block_id: str) -> None:
        if self._runner.is_running:
            return
        block = self._program.blocks.get(block_id)
        position = self._handle_offset(block_id)
        if block is None or position is None:
            return
        if self._drag is None or self._drag["id"] != block_id:
            if self._marquee_active:
                self._marquee_cancelled = True  # moving a block, not selecting
                if self._marquee_rect is not None:
                    self._marquee_rect.visible = False
            # First movement of a new drag: decide what moves and remember
            # the start positions. A selected block drags the whole selection.
            if block_id in self._selected and len(self._selected) > 1:
                roots = self._selection_roots()
            else:
                if block_id not in self._selected:
                    self._set_selection(set())
                roots = [block_id]
            chain: list[str] = []
            for root in roots:  # never takes an enclosing C's arm
                chain.extend(b for b in self._program.segment_ids(root) if b not in chain)
            primary = next(r for r in roots if block_id in self._program.segment_ids(r))
            self._drag = {
                "id": block_id,
                "roots": roots,
                "primary": primary,
                "start": {b_id: (self._program.get(b_id).x, self._program.get(b_id).y)
                          for b_id in chain},
            }
            print(f"[Scratch Blocks][drag] start {block.type}: {len(roots)} group(s), "
                  f"{len(chain)} block(s)")
            asyncio.ensure_future(self._watch_drag_release(block_id))
        dx = position[0] - block.x
        dy = position[1] - block.y
        if self._drag.get("last") == (dx, dy):
            return  # x and y callbacks fire separately; move once per position
        self._drag["last"] = (dx, dy)
        for b_id, (start_x, start_y) in self._drag["start"].items():
            visual = self._block_placers.get(b_id)
            if visual is not None:
                visual.offset_x = ui.Pixel(self._visual_offset(start_x + dx))
                visual.offset_y = ui.Pixel(self._visual_offset(start_y + dy))

    @staticmethod
    def _left_mouse_down() -> bool | None:
        """True while the left mouse button is held; None if it can't be read."""
        try:
            import omni.appwindow
            mouse = omni.appwindow.get_default_app_window().get_mouse()
            value = carb.input.acquire_input_interface().get_mouse_value(
                mouse, carb.input.MouseInput.LEFT_BUTTON)
            return float(value) > 0.5
        except Exception as error:
            print(f"[Scratch Blocks][drag] cannot read mouse button: {error}")
            return None

    async def _watch_drag_release(self, block_id: str) -> None:
        """Poll the mouse once per frame and finish the drag on release."""
        import omni.kit.app
        app = omni.kit.app.get_app()
        while True:
            await app.next_update_async()
            if self._drag is None or self._drag["id"] != block_id:
                return  # canvas rebuilt or another drag took over
            state = self._left_mouse_down()
            if state is False:
                break
            if state is None:
                # Cannot read the button: wait until the handle stops moving.
                last = self._handle_offset(block_id)
                for _ in range(15):
                    await app.next_update_async()
                if self._handle_offset(block_id) != last:
                    continue
                break
        self._finish_drag(block_id)

    def _finish_drag(self, block_id: str) -> None:
        drag, self._drag = self._drag, None
        block = self._program.blocks.get(block_id)
        position = self._handle_offset(block_id)
        if drag is None or block is None or position is None or self._runner.is_running:
            self._refresh_canvas()
            return
        dx, dy = position[0] - block.x, position[1] - block.y
        if math.hypot(dx, dy) < DRAG_CLICK_TOLERANCE:
            self._refresh_canvas()  # a click, not a move: restore exact positions
            return
        start = drag["start"]
        primary = drag.get("primary", block_id)
        try:
            # Every group keeps its shape: all roots move by the same offset,
            # measured from the positions recorded when the drag started.
            targets = {}
            for root in drag.get("roots", [block_id]):
                root_x, root_y = start[root]
                targets[root] = (max(0.0, min(CANVAS_WIDTH - BLOCK_WIDTH, root_x + dx)),
                                 max(0.0, min(CANVAS_HEIGHT - BLOCK_HEIGHT, root_y + dy)))
            for root, (new_x, new_y) in targets.items():
                self._program.move_segment(root, new_x, new_y)
            # Only the group under the mouse snaps onto other blocks.
            x, y = targets[primary]
            lead = self._program.get(primary)
            lead.x, lead.y = x, y  # keep the drop point for snapping
            insertion = self._nearest_insertion(x, y, primary)
            if insertion is not None:
                mode, target_id = insertion
                if mode == "before":
                    self._program.connect_before(target_id, primary)
                else:
                    self._program.connect_after(target_id, primary)
                self._set_status(f"Connected {mode} {self._block_name(target_id)}")
            else:
                self._set_status(f"Placed at ({x:.0f}, {y:.0f})")
            print(f"[Scratch Blocks][drag] released at ({x:.0f}, {y:.0f}); snap={insertion}")
        except Exception as error:
            self._set_status(f"Move failed: {error}")
            print(f"[Scratch Blocks][drag] move failed: {error}")
        self._refresh_canvas()

    # ------------------------------------------------------------------
    # Box selection (like the Windows desktop): press on empty canvas, drag a
    # box, release. Selected blocks get a blue outline and move together.
    # ------------------------------------------------------------------
    @staticmethod
    def _sized_rect(offset_x: float, offset_y: float, width: float, height: float,
                    style: dict | None = None, **rect_kwargs: Any) -> tuple[Any, Any, Any]:
        """Rectangle with an explicit pixel size at every level.

        In Isaac Sim 6.2 a Rectangle inside a Placer stretched to fill the
        canvas (REPEAT bar, selection outlines) unless the Placer and its
        container also have explicit sizes. Returns (placer, stack, rect).
        """
        w, h = ui.Pixel(width), ui.Pixel(height)
        placer = ui.Placer(offset_x=ui.Pixel(offset_x), offset_y=ui.Pixel(offset_y),
                           width=w, height=h, stable_size=True,
                           **({"style": style} if style is not None else {}))
        with placer:
            stack = ui.ZStack(width=w, height=h)
            with stack:
                rect = ui.Rectangle(width=w, height=h, **rect_kwargs)
        return placer, stack, rect

    def _build_selection_outline(self, block: ProgramBlock, width: float) -> None:
        """Blue frame around a selected block, shown/hidden without rebuilds."""
        scale = self._zoom
        _, _, self._select_rects[block.id] = self._sized_rect(
            -3 * scale, -3 * scale, (width + 6) * scale, (BLOCK_HEIGHT + 2) * scale,
            opaque_for_mouse_events=False,
            visible=(block.id in self._selected),
            style={"border_color": SELECT_COLOR, "border_width": 2,
                   "background_color": 0x00000000},
        )

    def _build_marquee(self) -> None:
        """The selection box itself: built once per canvas, hidden until used.

        Margin 0 so its corners sit exactly under the mouse (the window style
        would otherwise shift it by its inherited margin).
        """
        self._marquee_placer, self._marquee_stack, self._marquee_rect = self._sized_rect(
            0.0, 0.0, 1.0, 1.0, style={"margin": 0},
            visible=False, opaque_for_mouse_events=False,
        )
        self._marquee_rect.set_style({"margin": 0, "background_color": SELECT_FILL,
                                      "border_color": SELECT_COLOR, "border_width": 1})

    def _set_selection(self, ids: set[str]) -> None:
        self._selected = set(ids)
        for block_id, rect in self._select_rects.items():
            rect.visible = block_id in self._selected

    def _selection_roots(self) -> list[str]:
        """Top selected blocks of each group; blocks already carried along
        below another selected block are not separate groups."""
        chain_order = []
        if self._program.start_next is not None:
            chain_order = self._program.chain_ids(self._program.start_next)
        order = {b_id: i for i, b_id in enumerate(chain_order)}
        candidates = sorted(
            (b for b in self._selected
             if b in self._program.blocks and self._program.get(b).type != "end_repeat"),
            key=lambda b: (order.get(b, len(order)), self._program.get(b).y),
        )
        roots: list[str] = []
        covered: set[str] = set()
        for candidate in candidates:
            if candidate in covered:
                continue
            roots.append(candidate)
            covered.update(self._program.segment_ids(candidate))
        return roots

    def _blocks_in_rect(self, x0: float, y0: float, x1: float, y1: float) -> set[str]:
        """Blocks whose coloured body overlaps a logical canvas rectangle.

        END REPEAT arms are never selected on their own; a selected REPEAT
        already carries its arm along.
        """
        left, right = min(x0, x1), max(x0, x1)
        top, bottom = min(y0, y1), max(y0, y1)
        found = set()
        for block in self._program.blocks.values():
            if block.type == "end_repeat":
                continue
            bx0, bx1 = block.x, block.x + self._width_for_block(block)
            by0, by1 = block.y + HANDLE_TOP, block.y + HANDLE_TOP + HANDLE_HEIGHT
            if bx0 < right and bx1 > left and by0 < bottom and by1 > top:
                found.add(block.id)
        return found

    def _blocks_in_screen_rect(self, a: tuple[float, float], b: tuple[float, float]) -> set[str]:
        """Blocks whose drawn body overlaps the selection box, in screen units.

        Uses each block image's real on-screen position and size, so it does
        not depend on margins, zoom or scrolling. Falls back to the logical
        test if images are unavailable.
        """
        left, right = min(a[0], b[0]), max(a[0], b[0])
        top, bottom = min(a[1], b[1]), max(a[1], b[1])
        found, missing = set(), False
        for block in self._program.blocks.values():
            if block.type == "end_repeat":
                continue
            image = self._block_images.get(block.id)
            try:
                ix, iy = float(image.screen_position_x), float(image.screen_position_y)
                iw, ih = float(image.computed_width), float(image.computed_height)
            except Exception:
                missing = True
                continue
            # Coloured body is the middle band of the image (about 14..44 of 52).
            body_top, body_bottom = iy + ih * 14.0 / BLOCK_HEIGHT, iy + ih * 44.0 / BLOCK_HEIGHT
            if ix < right and ix + iw > left and body_top < bottom and body_bottom > top:
                found.add(block.id)
        if missing:
            origin = self._canvas_stack or self._canvas_surface
            ox = float(origin.screen_position_x) if origin is not None else 0.0
            oy = float(origin.screen_position_y) if origin is not None else 0.0
            found |= self._blocks_in_rect((left - ox) / self._zoom, (top - oy) / self._zoom,
                                          (right - ox) / self._zoom, (bottom - oy) / self._zoom)
        return found

    def _canvas_pressed(self, x: float, y: float, button: Any, modifiers: Any) -> None:
        """Press on empty canvas: start a selection box."""
        try:
            if int(button) not in (0, 1):
                return
        except (TypeError, ValueError):
            pass
        if self._marquee_active or self._runner.is_running or self._drag is not None:
            return
        # omni.ui sends a press to EVERY widget under the mouse, so grabbing a
        # block also reaches this canvas surface. Only empty canvas starts a box.
        surface = self._canvas_surface
        origin_x = float(surface.screen_position_x) if surface is not None else 0.0
        origin_y = float(surface.screen_position_y) if surface is not None else 0.0
        if self._point_on_block((float(x) - origin_x) / self._zoom,
                                (float(y) - origin_y) / self._zoom):
            return
        self._remember_press(x, y)  # press point in ui and raw mouse pixels
        if self._press is None:
            print("[Scratch Blocks][select] cannot read mouse position; box selection off")
            return
        # Measure how many window pixels one ui unit is, from this very press,
        # instead of trusting the reported DPI factor.
        _, _, raw_x, raw_y = self._press
        dpi = self._pixels_per_ui_unit()
        sx = raw_x / float(x) if abs(float(x)) > 50 else dpi
        sy = raw_y / float(y) if abs(float(y)) > 50 else dpi
        self._mouse_scale = (sx, sy) if 0.5 < sx < 4 and 0.5 < sy < 4 else (dpi, dpi)
        print(f"[Scratch Blocks][select] press ui=({float(x):.0f},{float(y):.0f}) "
              f"raw=({raw_x:.0f},{raw_y:.0f}) reported dpi={dpi:.3f} "
              f"measured=({self._mouse_scale[0]:.3f},{self._mouse_scale[1]:.3f})")
        self._marquee_active = True
        self._marquee_cancelled = False
        # The old selection stays until the box is released (or a click on
        # empty canvas clears it), so a press can never silently drop it.
        asyncio.ensure_future(self._track_marquee())

    def _point_on_block(self, lx: float, ly: float, pad: float = 4.0) -> bool:
        """True if a logical canvas point is on START, a block, or a REPEAT bar."""
        if START_X - pad <= lx <= START_X + 160 + pad and START_Y - pad <= ly <= START_Y + 58 + pad:
            return True
        for block in self._program.blocks.values():
            left, right = block.x - pad, block.x + self._width_for_block(block) + pad
            top, bottom = block.y + HANDLE_TOP - pad, block.y + HANDLE_TOP + HANDLE_HEIGHT + pad
            if left <= lx <= right and top <= ly <= bottom:
                return True
            if block.type == "repeat":
                end = self._program.matching_end(block.id)
                if end is not None and block.x - pad <= lx <= block.x + REPEAT_INDENT + pad \
                        and block.y + HANDLE_TOP <= ly <= end.y + HANDLE_TOP + HANDLE_HEIGHT:
                    return True
        return False

    def _mouse_ui_position(self) -> tuple[float, float] | None:
        """Current mouse position in omni.ui coordinates, from the stored press."""
        raw = self._carb_mouse_pixels()
        if self._press is None or raw is None:
            return None
        press_x, press_y, raw_x0, raw_y0 = self._press
        sx, sy = self._mouse_scale or (self._pixels_per_ui_unit(),) * 2
        return press_x + (raw[0] - raw_x0) / sx, press_y + (raw[1] - raw_y0) / sy

    async def _track_marquee(self) -> None:
        import omni.kit.app
        app = omni.kit.app.get_app()
        start_ui = (self._press[0], self._press[1]) if self._press else (0.0, 0.0)
        current = start_ui
        try:
            while True:
                await app.next_update_async()
                position = self._mouse_ui_position()
                if position is not None:
                    current = position
                    self._draw_marquee(start_ui, current)
                if self._marquee_cancelled or self._left_mouse_down() is not True:
                    break
        finally:
            self._marquee_active = False
            if self._marquee_rect is not None:
                self._marquee_rect.visible = False
        if self._marquee_cancelled:
            return  # a block drag took over this press
        if math.hypot(current[0] - start_ui[0], current[1] - start_ui[1]) < MARQUEE_CLICK_TOLERANCE:
            self._set_selection(set())
            self._set_status("Selection cleared")
            return
        selected = self._blocks_in_screen_rect(start_ui, current)
        self._set_selection(selected)
        self._set_status(f"{len(selected)} block(s) selected - drag any of them to move the group")
        print(f"[Scratch Blocks][select] {len(selected)} block(s)")

    def _draw_marquee(self, start: tuple[float, float], current: tuple[float, float]) -> None:
        container = self._canvas_stack or self._canvas_surface
        placer, rect = self._marquee_placer, self._marquee_rect
        if container is None or placer is None or rect is None:
            return
        origin_x, origin_y = float(container.screen_position_x), float(container.screen_position_y)
        left, top = min(start[0], current[0]) - origin_x, min(start[1], current[1]) - origin_y
        width = ui.Pixel(max(1.0, abs(current[0] - start[0])))
        height = ui.Pixel(max(1.0, abs(current[1] - start[1])))
        placer.offset_x = ui.Pixel(max(0.0, left))
        placer.offset_y = ui.Pixel(max(0.0, top))
        for widget in (placer, self._marquee_stack, rect):
            if widget is not None:
                widget.width, widget.height = width, height
        rect.visible = True

    async def _refresh_next_frame(self) -> None:
        import omni.kit.app
        await omni.kit.app.get_app().next_update_async()
        self._refresh_canvas()

    def _build_start(self) -> None:
        with ui.Placer(
            offset_x=ui.Pixel(START_X * self._zoom),
            offset_y=ui.Pixel(START_Y * self._zoom),
            width=ui.Pixel(160 * self._zoom), height=ui.Pixel(58 * self._zoom),
            stable_size=True,
        ):
            with ui.ZStack(
                width=ui.Pixel(160 * self._zoom),
                height=ui.Pixel(58 * self._zoom),
                accept_drop_fn=self._accept_drop,
                drop_fn=self._drop_on_canvas,
                mouse_wheel_fn=self._on_canvas_wheel,
            ):
                ui.Image(
                    os.path.join(self._asset_dir, "start_block.svg"),
                    width=ui.Pixel(160 * self._zoom),
                    height=ui.Pixel(58 * self._zoom),
                    fill_policy=ui.FillPolicy.STRETCH,
                    alignment=ui.Alignment.LEFT_TOP,
                    accept_drop_fn=self._accept_drop,
                    drop_fn=self._drop_on_canvas,
                )
                ui.Label(
                    "START",
                    width=ui.Pixel(160 * self._zoom),
                    height=ui.Pixel(58 * self._zoom),
                    alignment=ui.Alignment.CENTER,
                    accept_drop_fn=self._accept_drop,
                    drop_fn=self._drop_on_canvas,
                    style={"color": 0xFFFFFFFF, "font_size": max(8, int(13 * self._zoom))},
                )

    # MOVE ALL JOINTS: one editable degree field for every selected robot joint.
    def _all_joint_names(self, block: ProgramBlock) -> list[str]:
        selection = self._robot_for_joint_block(block.id)
        if selection is not None:
            detected = self._cached_joint_names(selection["robot_path"])
            if detected:
                # Use the configured arm-joint order for known robots, and
                # detected rotational joints for a generic articulation.
                profile = self._cached_profile(selection["profile"])
                configured = list(profile.get("arm_joint_names", []))
                if configured and set(configured).issubset(set(detected)):
                    return configured
                return detected
        # Saved programs remain legible even when the USD stage is not open.
        try:
            return list(parse_joint_targets(block.value.get("targets", "")))
        except (ValueError, TypeError):
            return []

    # ------------------------------------------------------------------
    # Step 1 (performance): cached stage scan and profile lookups
    # ------------------------------------------------------------------
    def _cached_joint_names(self, robot_path: str) -> list[str]:
        """Rotational joints of a robot; the USD stage is scanned once per path."""
        names = self._joint_cache.get(robot_path)
        if names is None:
            from .robot_scan import revolute_joint_names_of
            names = revolute_joint_names_of(robot_path)
            self._joint_cache[robot_path] = names
        return list(names)

    def _cached_profile(self, name: str) -> dict:
        """Robot profile; robot_profiles.json is read once per profile name.

        Treat the returned dict as read-only.
        """
        profile = self._profile_cache.get(name)
        if profile is None:
            from .robot_profiles import load_profile
            profile = load_profile(name)
            self._profile_cache[name] = profile
        return profile

    def _invalidate_scene_cache(self) -> None:
        """Forget cached joints/profiles, e.g. after a robot or stage change."""
        self._joint_cache.clear()
        self._profile_cache.clear()

    def _all_joint_targets(self, block: ProgramBlock) -> dict[str, float]:
        try:
            return parse_joint_targets(block.value.get("targets", ""))
        except (ValueError, TypeError):
            return {}

    def _sync_all_joint_targets(self, block: ProgramBlock, names: list[str]) -> dict[str, float]:
        old = self._all_joint_targets(block)
        values = {name: old.get(name, 0.0) for name in names}
        # Keep the original value representation for compatibility with the runner.
        if names and list(old) != names and not self._runner.is_running:
            block.value = {"targets": ", ".join(f"{name}={value:g}" for name, value in values.items())}
        return values

    def _width_for_block(self, block: ProgramBlock) -> float:
        if block.type == "end_repeat":
            return END_ARM_WIDTH
        if block.type == "wave":
            return WAVE_WIDTH
        if block.type != "move_all_joints":
            return BLOCK_WIDTH
        count = len(self._all_joint_names(block))
        # 120px title, 76px per joint (label + edit box), 62px for delete.
        return max(float(BLOCK_WIDTH), 120.0 + 76.0 * count + 62.0)

    def _build_all_joint_fields(self, block: ProgramBlock) -> None:
        names = self._all_joint_names(block)
        if not names:
            scale = self._zoom
            with ui.Placer(offset_x=ui.Pixel((block.x + 124) * scale),
                           offset_y=ui.Pixel((block.y + WIDGET_TOP) * scale),
                           width=ui.Pixel(170 * scale), height=ui.Pixel(WIDGET_HEIGHT * scale),
                           stable_size=True):
                ui.Label("INIT ROBOT first", width=ui.Pixel(170 * scale),
                         height=ui.Pixel(WIDGET_HEIGHT * scale),
                         opaque_for_mouse_events=False,
                         style={"color": 0xFFFFFFFF, "font_size": max(8, int(12 * scale))})
            return
        values = self._sync_all_joint_targets(block, names)
        scale = self._zoom
        for index, name in enumerate(names):
            key = (block.id, name)
            model = self._all_joint_models.get(key)
            if model is None:
                model = ui.SimpleStringModel(f"{values[name]:g}")
                self._all_joint_models[key] = model
                model.add_end_edit_fn(
                    lambda changed, b_id=block.id, j_name=name:
                        self._all_joint_angle_edited(b_id, j_name, changed)
                )
            # Each joint is a fixed-width horizontal group, kept fully inside
            # the coloured part of the Scratch block at every canvas zoom.
            group_left = (block.x + 120 + index * 76) * scale
            group_top = (block.y + WIDGET_TOP) * scale
            label_width = ui.Pixel(37 * scale)
            edit_width = ui.Pixel(36 * scale)
            field_height = ui.Pixel(WIDGET_HEIGHT * scale)

            with ui.Placer(offset_x=ui.Pixel(group_left),
                           offset_y=ui.Pixel(group_top),
                           width=ui.Pixel(74 * scale), height=field_height,
                           stable_size=True):
                with ui.HStack(width=ui.Pixel(74 * scale),
                               height=field_height, spacing=0,
                               style={"margin": 0}):
                    ui.Label(f"Joint {index + 1}", width=label_width,
                             height=field_height,
                             alignment=ui.Alignment.CENTER,
                             tooltip=name, opaque_for_mouse_events=False,
                             style={"color": 0xFFFFFFFF,
                                    "font_size": max(7, int(9 * scale)),
                                    "margin": 0})
                    field = ui.StringField(
                        model=model, width=edit_width, height=field_height,
                        enabled=True,
                        style={"color": 0xFF1A1A1A, "font_size": max(8, int(11 * scale)), "margin": 0},
                        tooltip=f"{name}: Zielwinkel in Grad; Enter zum Speichern",
                        mouse_pressed_fn=lambda x, y, button, mods, b_id=block.id, j_name=name:
                            self._all_joint_field_pressed(b_id, j_name, button),
                    )
                    self._all_joint_fields[key] = field

    @staticmethod
    def _focus_now(field: Any) -> None:
        """Give a text field keyboard focus in the same click.

        The old next-frame focus was needed while blocks were drag-and-drop
        sources. Blocks now move via their title handle, so focusing directly
        works; the next-frame call stays as a fallback only.
        """
        if field is None:
            return
        try:
            field.focus_keyboard(True)
        except Exception:
            pass

    def _all_joint_field_pressed(self, block_id: str, joint_name: str, button: Any) -> None:
        if self._runner.is_running:
            return
        try:
            if int(button) not in (0, 1):
                return
        except (TypeError, ValueError):
            pass
        self._focus_now(self._all_joint_fields.get((block_id, joint_name)))
        asyncio.ensure_future(self._focus_all_joint_next_frame(block_id, joint_name))

    async def _focus_all_joint_next_frame(self, block_id: str, joint_name: str) -> None:
        import omni.kit.app
        await omni.kit.app.get_app().next_update_async()
        field = self._all_joint_fields.get((block_id, joint_name))
        if field is not None:
            try:
                field.focus_keyboard(True)
            except Exception as exc:
                print(f"[Scratch Blocks] MOVE ALL keyboard focus: {exc}")

    def _all_joint_angle_edited(self, block_id: str, joint_name: str, model: Any) -> None:
        block = self._program.blocks.get(block_id)
        if block is None or block.type != "move_all_joints" or self._runner.is_running:
            return
        try:
            degrees = float(model.get_value_as_string().strip().replace(",", "."))
            if not math.isfinite(degrees):
                raise ValueError("Winkel muss endlich sein")
            names = self._all_joint_names(block)
            if joint_name not in names:
                raise ValueError("Gelenk gehört nicht zum ausgewählten Roboter")
            values = self._all_joint_targets(block)
            values[joint_name] = degrees
            block.value = {"targets": ", ".join(f"{name}={values.get(name, 0.0):g}" for name in names)}
            self._set_status(f"MOVE ALL: {joint_name} = {degrees:g}°")
        except (ValueError, TypeError) as exc:
            self._set_status(f"MOVE ALL: ungültiger Winkel für {joint_name}: {exc}")

    def _build_block(self, block: ProgramBlock) -> None:
        # Step 2: canvas blocks are moved by their drag handle, not by
        # drag-and-drop, so the block surface carries no drag payload.
        asset = {
            "print": "print_block.svg",
            "wait": "wait_block.svg",
            "initialize_robot": "initialize_robot_block.svg",
            "initialize_gripper": "initialize_robot_block.svg",
            "robot_properties": "robot_properties_block.svg",
            "set_speed": "move_joint_block.svg",
            "move_all_joints": "move_joint_block.svg",
            "move_joint": "move_joint_block.svg",
            "move_linear": "move_linear_block.svg",
            "move_relative": "move_linear_block.svg",
            "wave": "move_joint_block.svg",
            "gripper": "gripper_block.svg",
            "repeat": "repeat_block.svg",
            "end_repeat": "repeat_block.svg",
            "stop_program": "stop_program_block.svg",
        }[block.type]
        block_width = self._width_for_block(block)
        if block.type == "move_all_joints":
            joint_count = min(len(self._all_joint_names(block)), 24)
            asset = f"move_all_joints_{joint_count}.svg"
        with ui.Placer(
            offset_x=ui.Pixel(block.x * self._zoom),
            offset_y=ui.Pixel(block.y * self._zoom),
            width=ui.Pixel(block_width * self._zoom),
            height=ui.Pixel(BLOCK_HEIGHT * self._zoom),
            stable_size=True,
        ):
            with ui.ZStack(
                width=ui.Pixel(block_width * self._zoom),
                height=ui.Pixel(BLOCK_HEIGHT * self._zoom),
                accept_drop_fn=self._accept_drop,
                drop_fn=self._drop_on_canvas,
                mouse_wheel_fn=self._on_canvas_wheel,
            ):
                self._block_images[block.id] = ui.Image(
                    os.path.join(self._asset_dir, asset),
                    width=ui.Pixel(block_width * self._zoom),
                    height=ui.Pixel(BLOCK_HEIGHT * self._zoom),
                    fill_policy=ui.FillPolicy.STRETCH,
                    alignment=ui.Alignment.LEFT_TOP,
                    accept_drop_fn=self._accept_drop,
                    drop_fn=self._drop_on_canvas,
                )
                with ui.HStack(
                    height=ui.Pixel(44 * self._zoom),
                    spacing=max(2, int(6 * self._zoom)),
                    alignment=ui.Alignment.CENTER,
                    style={"margin": max(2, int(6 * self._zoom))},
                ):
                    ui.Label(
                        {
                            "print": "PRINT",
                            "wait": "WAIT",
                            "initialize_robot": "INIT ROBOT",
                            "initialize_gripper": "INIT GRIPPER",
                            "robot_properties": "ROBOT PROPERTIES",
                            "set_speed": "SPEED °/s",
                            "move_all_joints": "MOVE ALL JOINTS",
                            "move_joint": "SINGLE JOINT",
                            "move_linear": "MOVE LINEAR",
                            "move_relative": "MOVE RELATIVE",
                            "wave": "WAVE",
                            "gripper": "GRIPPER",
                            "repeat": "REPEAT",
                            "end_repeat": "",
                            "stop_program": "STOP PROGRAM",
                        }[block.type],
                        width=ui.Pixel(self._label_width(block) * self._zoom),
                        accept_drop_fn=self._accept_drop,
                        drop_fn=self._drop_on_canvas,
                        style={"color": 0xFFFFFFFF, "font_size": max(8, int((10 if block.type == "move_relative" else 13) * self._zoom))},
                    )
                    # Keep original layout widths but draw the input widgets
                    # as foreground siblings, not inside the drag-enabled block.
                    if block.type in ("end_repeat", "stop_program", "wave"):
                        pass  # WAVE: its fields are drawn on top in _build_wave_fields
                    elif block.type == "robot_properties":
                        pass  # Dieser Block besitzt keine Eingabefelder.
                    elif block.type in ("move_all_joints", "move_joint", "move_linear", "move_relative", "gripper"):
                        ui.Spacer(width=ui.Pixel(175 * self._zoom))
                    elif block.type in ("print", "initialize_robot", "initialize_gripper"):
                        slot_width = {"initialize_robot": 185, "initialize_gripper": 175}.get(block.type, 220)
                        ui.Spacer(width=ui.Pixel(slot_width * self._zoom))
                    else:
                        ui.Spacer(width=ui.Pixel(65 * self._zoom))
                        ui.Label(
                            "deg/s" if block.type == "set_speed" else "times" if block.type == "repeat" else "seconds",
                            width=ui.Pixel(58 * self._zoom),
                            style={"color": 0xFFFFFFFF, "font_size": max(8, int(13 * self._zoom))},
                        )
                    # Fixed gap: the flexible Spacer() pushed the delete button
                    # outside the colored block when the canvas was wide.
                    ui.Spacer(
                        width=ui.Pixel((12 if block.type == "print" else 0 if block.type in ("initialize_robot", "initialize_gripper") else 104) * self._zoom)
                    )
                    # Reserve button space; clickable button is drawn separately
                    # above all blocks in _build_canvas().
                    ui.Spacer(width=ui.Pixel(36 * self._zoom))

        # Step 1 (performance): every block gets its highlight frame now,
        # hidden by default. _set_active_block only toggles visibility, so a
        # running program no longer rebuilds the whole canvas per block.
        _, _, self._highlight_rects[block.id] = self._sized_rect(
            (block.x - 3) * self._zoom, (block.y - 3) * self._zoom,
            (block_width + 6) * self._zoom, (BLOCK_HEIGHT + 2) * self._zoom,
            opaque_for_mouse_events=False,
            visible=(block.id == self._active_block_id),
            style={"border_color": 0xFFFFFF00, "border_width": 3,
                   "background_color": 0x00000000},
        )

    def _build_repeat_field(self, block: ProgramBlock) -> None:
        """Inline loop count, editable like SET SPEED."""
        scale = self._zoom
        width = ui.Pixel(74 * scale)
        height = ui.Pixel(WIDGET_HEIGHT * scale)
        model = self._repeat_models.get(block.id)
        if model is None:
            model = ui.SimpleStringModel(f"{int(block.value)}")
            self._repeat_models[block.id] = model
            model.add_end_edit_fn(lambda changed, block_id=block.id: self._repeat_edited(block_id, changed))
        with ui.Placer(offset_x=ui.Pixel((block.x + 112) * scale),
                       offset_y=ui.Pixel((block.y + WIDGET_TOP) * scale),
                       width=width, height=height, stable_size=True):
            field = ui.StringField(
                model=model, width=width, height=height, enabled=True,
                style={"color": 0xFF1A1A1A, "font_size": max(9, int(12 * scale)), "margin": 0},
                tooltip="Anzahl Wiederholungen (1 bis 100); mit Enter bestaetigen",
                mouse_pressed_fn=lambda x, y, button, mods, block_id=block.id:
                    self._repeat_pressed(block_id, button),
            )
            self._repeat_fields[block.id] = field

    def _repeat_pressed(self, block_id: str, button: Any) -> None:
        if self._runner.is_running or block_id not in self._program.blocks:
            return
        self._focus_now(self._repeat_fields.get(block_id))
        asyncio.ensure_future(self._repeat_focus_next_frame(block_id))

    async def _repeat_focus_next_frame(self, block_id: str) -> None:
        import omni.kit.app
        await omni.kit.app.get_app().next_update_async()
        field = self._repeat_fields.get(block_id)
        if field is not None and block_id in self._program.blocks:
            field.focus_keyboard(True)

    def _repeat_edited(self, block_id: str, model: Any) -> None:
        block = self._program.blocks.get(block_id)
        if block is None or block.type != "repeat" or self._runner.is_running:
            return
        try:
            number = float(model.get_value_as_string().strip())
            if not math.isfinite(number) or not number.is_integer() or not 1 <= number <= 100:
                raise ValueError("REPEAT: whole number from 1 to 100 required")
            block.value = number
            self._set_status(f"REPEAT: {int(number)} repetitions")
        except (TypeError, ValueError) as error:
            model.set_value(f"{int(block.value)}")
            self._set_status(str(error))

    def _build_speed_field(self, block: ProgramBlock) -> None:
        """Direct, keyboard-editable speed in degrees/second."""
        scale = self._zoom
        width = ui.Pixel(76 * scale)
        height = ui.Pixel(WIDGET_HEIGHT * scale)
        model = self._speed_models.get(block.id)
        if model is None:
            model = ui.SimpleStringModel(f"{float(block.value):g}")
            self._speed_models[block.id] = model
            model.add_end_edit_fn(
                lambda changed, block_id=block.id:
                    self._speed_edited(block_id, changed)
            )
        with ui.Placer(
            offset_x=ui.Pixel((block.x + 104) * scale),
            offset_y=ui.Pixel((block.y + WIDGET_TOP) * scale),
            width=width, height=height, stable_size=True,
        ):
            field = ui.StringField(
                model=model, width=width, height=height, enabled=True,
                style={"color": 0xFF1A1A1A, "font_size": max(9, int(12 * scale)), "margin": 0},
                tooltip="Gelenkgeschwindigkeit in Grad/Sekunde; Enter zum Speichern",
                mouse_pressed_fn=lambda x, y, button, mods, block_id=block.id:
                    self._speed_field_pressed(block_id, button),
            )
            self._speed_fields[block.id] = field

    def _speed_field_pressed(self, block_id: str, button: Any) -> None:
        if self._runner.is_running or block_id not in self._program.blocks:
            return
        try:
            if int(button) not in (0, 1):
                return
        except (TypeError, ValueError):
            pass
        self._focus_now(self._speed_fields.get(block_id))
        asyncio.ensure_future(self._focus_speed_next_frame(block_id))

    async def _focus_speed_next_frame(self, block_id: str) -> None:
        import omni.kit.app
        await omni.kit.app.get_app().next_update_async()
        field = self._speed_fields.get(block_id)
        if field is not None and block_id in self._program.blocks:
            try:
                field.focus_keyboard(True)
            except Exception as error:
                self._set_status(f"SET SPEED: Fokusfehler: {error}")

    def _speed_edited(self, block_id: str, model: Any) -> None:
        block = self._program.blocks.get(block_id)
        if block is None or block.type != "set_speed" or self._runner.is_running:
            return
        try:
            speed = validate_speed(model.get_value_as_string().strip().replace(",", "."))
        except (ValueError, TypeError, OverflowError) as error:
            self._set_status(f"SET SPEED: {error}")
            # Restore the previous valid value after an invalid edit.
            model.set_value(f"{float(block.value):g}")
            return
        block.value = speed
        self._set_status(f"SET SPEED: {speed:g} deg/s")

    def _build_editable_field(self, block: ProgramBlock) -> None:
        """Clickable parameter display; editing happens in its own window.

        This deliberately avoids text focus inside the drag-enabled canvas.
        INIT ROBOT is the exception: it gets a robot menu directly on the block.
        """
        if block.type in ("robot_properties", "end_repeat", "repeat", "stop_program"):
            return
        if block.type == "wave":
            # Joint menu: same dropdown as SINGLE JOINT. Numbers: _build_wave_fields.
            self._build_move_joint_inline(block, (block.x + 62) * self._zoom,
                                          (block.y + WIDGET_TOP) * self._zoom)
            return
        scale = self._zoom
        is_init = block.type == "initialize_robot"
        if block.type == "initialize_gripper":
            left = (block.x + 6 + 100 + 6) * scale
            top = (block.y + WIDGET_TOP) * scale
            self._build_init_gripper_selector(block, left, top, ui.Pixel(175 * scale),
                                              ui.Pixel(WIDGET_HEIGHT * scale))
            return
        is_joint = block.type == "move_joint"
        is_all = block.type == "move_all_joints"
        is_linear = block.type in ("move_linear", "move_relative")
        is_gripper = block.type == "gripper"
        field_width = 175 if (is_all or is_joint or is_linear or is_gripper) else 65 if block.type in ("wait", "set_speed") else (185 if is_init else 220)
        label_width = 90 if (is_init or is_all or is_joint or is_linear or is_gripper or block.type == "set_speed") else 43
        left = (block.x + 6 + label_width + 6) * scale
        top = (block.y + WIDGET_TOP) * scale
        width = ui.Pixel(field_width * scale)
        height = ui.Pixel(WIDGET_HEIGHT * scale)
        on_press = lambda x, y, button, modifiers, block_id=block.id: self._parameter_pressed(block_id, button)
        if is_init:
            self._build_init_robot_selector(block, left, top, width, height)
            return
        if is_joint:
            self._build_move_joint_inline(block, left, top)
            return
        if is_gripper:
            self._build_gripper_selector(block, left, top)
            return
        if block.type == "set_speed":
            return  # The speed input is drawn above draggable block widgets.
        if is_all:
            return  # Joint captions and editable angle fields are rendered on top.
        if is_linear:
            return  # XYZ and the settings arrow are drawn as foreground controls.
        with ui.Placer(offset_x=ui.Pixel(left), offset_y=ui.Pixel(top),
                       width=width, height=height, stable_size=True):
            with ui.ZStack(width=width, height=height, style={"margin": 0}):  # margin 0: the window style {'margin': 7} would shrink these widgets
                ui.Rectangle(width=width, height=height,
                             style={"background_color": 0xFFFFFFFF, "border_radius": 10})
                display_value = (
                    f"{block.value['joint_name']} | {float(block.value['target_degrees']):g}°"
                    if is_joint else (
                        f"{block.value['axis'].upper()} | {float(block.value['distance']):g} m"
                        if is_linear else str(block.value)
                    )
                )
                if is_all:
                    display_value = str(block.value.get("targets", "")) or "Click: joint targets"
                ui.Label(display_value, width=width, height=height,
                         alignment=ui.Alignment.LEFT_CENTER,
                         mouse_pressed_fn=on_press,
                         tooltip="Click to edit parameter",
                         style={"color": 0xFF202020,
                                "font_size": max(8, int(13 * scale)),
                                "margin": 4})

    # ------------------------------------------------------------------
    # GRIPPER: inline Open / Close dropdown, no parameter window
    # ------------------------------------------------------------------
    def _build_gripper_selector(self, block: ProgramBlock, left: float, top: float) -> None:
        scale = self._zoom
        width = ui.Pixel(175 * scale)
        height = ui.Pixel(WIDGET_HEIGHT * scale)
        action = str(block.value).strip().lower()
        label = "Open" if action == "open" else "Close" if action == "close" else "Select action"
        press = lambda x, y, button, mods, block_id=block.id: self._gripper_field_pressed(block_id, button)
        drop = {"accept_drop_fn": self._accept_drop, "drop_fn": self._drop_on_canvas}
        with ui.Placer(offset_x=ui.Pixel(left), offset_y=ui.Pixel(top),
                       width=width, height=height, stable_size=True):
            with ui.ZStack(width=width, height=height, style={"margin": 0}, **drop):
                ui.Rectangle(width=width, height=height, mouse_pressed_fn=press, **drop,
                             style={"background_color": 0xFFFFFFFF, "border_radius": 7})
                with ui.HStack(width=width, height=height):
                    ui.Spacer(width=ui.Pixel(8 * scale))
                    self._gripper_labels[block.id] = ui.Label(label, height=height, alignment=ui.Alignment.LEFT_CENTER,
                             mouse_pressed_fn=press, **drop, tooltip="Select gripper action",
                             style={"color": 0xFF1A1A1A, "font_size": max(8, int(13 * scale))})
                    with ui.VStack(width=ui.Pixel(22 * scale), height=height):
                        ui.Spacer()
                        with ui.HStack(height=ui.Pixel(8 * scale)):
                            ui.Spacer()
                            ui.Triangle(width=ui.Pixel(10 * scale), height=ui.Pixel(7 * scale),
                                        alignment=ui.Alignment.CENTER_BOTTOM,
                                        mouse_pressed_fn=press,
                                        style={"background_color": 0xFF606060})
                            ui.Spacer()
                        ui.Spacer()

    def _gripper_field_pressed(self, block_id: str, button: Any) -> None:
        if self._runner.is_running or block_id not in self._program.blocks:
            return
        try:
            if int(button) not in (0, 1):
                return
        except (TypeError, ValueError):
            pass
        asyncio.ensure_future(self._show_gripper_menu_next_frame(block_id))

    async def _show_gripper_menu_next_frame(self, block_id: str) -> None:
        import omni.kit.app
        await omni.kit.app.get_app().next_update_async()
        block = self._program.blocks.get(block_id)
        if block is None or block.type != "gripper" or self._runner.is_running:
            return
        action = str(block.value).strip().lower()
        self._gripper_menu = ui.Menu("GRIPPER – Select action")
        with self._gripper_menu:
            for name, value in (("Open", "open"), ("Close", "close")):
                ui.MenuItem(name, checkable=True, checked=(action == value),
                            triggered_fn=lambda selected=value, b_id=block_id:
                                self._gripper_action_selected(b_id, selected))
        self._gripper_menu.show()

    def _gripper_action_selected(self, block_id: str, action: str) -> None:
        """Apply the choice now and change only the label text (no rebuild)."""
        block = self._program.blocks.get(block_id)
        label = self._gripper_labels.get(block_id)
        if block is None or block.type != "gripper" or self._runner.is_running \
                or action not in ("open", "close"):
            return
        if label is None:  # no label to update: fall back to the old rebuild path
            asyncio.ensure_future(self._apply_gripper_action_next_frame(block_id, action))
            return
        block.value = action
        label.text = "Open" if action == "open" else "Close"
        self._set_status(f"GRIPPER: {action}")

    async def _apply_gripper_action_next_frame(self, block_id: str, action: str) -> None:
        import omni.kit.app
        await omni.kit.app.get_app().next_update_async()
        block = self._program.blocks.get(block_id)
        if block is None or block.type != "gripper" or self._runner.is_running:
            return
        if action not in ("open", "close"):
            return
        block.value = action  # Same values expected by runner and compiler.
        self._refresh_canvas()
        self._set_status(f"GRIPPER: {action}")

    # ------------------------------------------------------------------
    # WAVE: joint menu (shared with SINGLE JOINT) + four inline number fields
    # ------------------------------------------------------------------
    def _build_wave_fields(self, block: ProgramBlock) -> None:
        """low / high / Hz / cycles, each a small caption + keyboard-editable field."""
        scale = self._zoom
        field_height = ui.Pixel(WIDGET_HEIGHT * scale)
        top = ui.Pixel((block.y + WIDGET_TOP) * scale)
        try:
            value = validate_wave_parameters(block.value)
        except (TypeError, ValueError):
            value = validate_wave_parameters({})
        for index, (key, caption, tip) in enumerate(WAVE_FIELDS):
            base = block.x + 170 + index * 70
            model_key = (block.id, key)
            model = self._wave_models.get(model_key)
            if model is None:
                model = ui.SimpleStringModel(f"{float(value[key]):g}")
                self._wave_models[model_key] = model
                model.add_end_edit_fn(
                    lambda changed, b_id=block.id, k=key: self._wave_edited(b_id, k, changed))
            with ui.Placer(offset_x=ui.Pixel(base * scale), offset_y=top,
                           width=ui.Pixel(28 * scale), height=field_height, stable_size=True):
                ui.Label(caption, width=ui.Pixel(28 * scale), height=field_height,
                         alignment=ui.Alignment.RIGHT_CENTER, opaque_for_mouse_events=False,
                         style={"color": 0xFFFFFFFF, "font_size": max(7, int(10 * scale)),
                                "margin": 0})
            width = ui.Pixel(38 * scale)
            with ui.Placer(offset_x=ui.Pixel((base + 31) * scale), offset_y=top,
                           width=width, height=field_height, stable_size=True):
                self._wave_fields[model_key] = ui.StringField(
                    model=model, width=width, height=field_height, enabled=True,
                    style={"color": 0xFF1A1A1A, "font_size": max(8, int(11 * scale)), "margin": 0},
                    tooltip=tip + "; Enter to save",
                    mouse_pressed_fn=lambda x, y, button, mods, b_id=block.id, k=key:
                        self._wave_field_pressed(b_id, k, button),
                )

    def _wave_field_pressed(self, block_id: str, key: str, button: Any) -> None:
        if self._runner.is_running or block_id not in self._program.blocks:
            return
        try:
            if int(button) not in (0, 1):
                return
        except (TypeError, ValueError):
            pass
        self._focus_now(self._wave_fields.get((block_id, key)))
        asyncio.ensure_future(self._focus_wave_next_frame(block_id, key))

    async def _focus_wave_next_frame(self, block_id: str, key: str) -> None:
        import omni.kit.app
        await omni.kit.app.get_app().next_update_async()
        field = self._wave_fields.get((block_id, key))
        if field is not None and block_id in self._program.blocks:
            try:
                field.focus_keyboard(True)
            except Exception as error:
                self._set_status(f"WAVE: focus error: {error}")

    def _wave_edited(self, block_id: str, key: str, model: Any) -> None:
        block = self._program.blocks.get(block_id)
        if block is None or block.type != "wave" or self._runner.is_running:
            return
        try:
            number = float(model.get_value_as_string().strip().replace(",", "."))
            updated = dict(block.value)
            updated[key] = number
            block.value = validate_wave_parameters(updated)
            self._set_status(f"WAVE: {key} = {number:g}")
        except (TypeError, ValueError, OverflowError) as error:
            model.set_value(f"{float(block.value.get(key, 0.0)):g}")   # restore the last valid value
            self._set_status(f"WAVE: {error}")

    # ------------------------------------------------------------------
    # MOVE LINEAR: inline X/Y/Z relative world displacements (metres)
    # ------------------------------------------------------------------
    def _build_linear_advanced_button(self, block: ProgramBlock) -> None:
        """Clickable dropdown arrow for the additional MOVE LINEAR parameters.

        Positioned after the Z field (ending at x=276) and before delete X
        (starting at x=300), and scaled with canvas zoom.
        """
        scale = self._zoom
        width = ui.Pixel(18 * scale)
        height = ui.Pixel(WIDGET_HEIGHT * scale)
        on_press = lambda x, y, button, mods, b_id=block.id: self._parameter_pressed(b_id, button)
        with ui.Placer(
            offset_x=ui.Pixel((block.x + 280) * scale),
            offset_y=ui.Pixel((block.y + WIDGET_TOP) * scale),
            width=width, height=height, stable_size=True,
        ):
            with ui.ZStack(width=width, height=height, style={"margin": 0}):
                ui.Rectangle(
                    width=width, height=height,
                    mouse_pressed_fn=on_press,
                    tooltip="Erweiterte MOVE LINEAR Einstellungen öffnen",
                    style={"background_color": 0xFF155A75, "border_radius": 4},
                )
                with ui.VStack(width=width, height=height):
                    ui.Spacer()
                    with ui.HStack(height=ui.Pixel(8 * scale)):
                        ui.Spacer()
                        ui.Triangle(
                            width=ui.Pixel(10 * scale),
                            height=ui.Pixel(7 * scale),
                            alignment=ui.Alignment.CENTER_BOTTOM,
                            mouse_pressed_fn=on_press,
                            tooltip="Schritte, Dauer und Kinematik einstellen",
                            style={"background_color": 0xFFFFFFFF},
                        )
                        ui.Spacer()
                    ui.Spacer()

    def _build_linear_fields(self, block: ProgramBlock) -> None:
        """Three evenly spaced label/input pairs, fully inside the block.

        Coordinates use the block's logical 360px width and are scaled together
        with the canvas. Each group has 59px: axis label, gap, numeric field.
        """
        scale = self._zoom
        group_start = 101
        group_step = 59
        label_width = 12
        label_to_field_gap = 4
        field_width = 41
        field_height = ui.Pixel(WIDGET_HEIGHT * scale)
        top = ui.Pixel((block.y + WIDGET_TOP) * scale)

        for index, axis in enumerate(("x", "y", "z")):
            base = block.x + group_start + index * group_step
            key = (block.id, axis)
            model = self._linear_models.get(key)
            if model is None:
                value = validate_linear_parameters(block.value)
                model = ui.SimpleStringModel(f"{float(value[axis]):g}")
                self._linear_models[key] = model
                model.add_end_edit_fn(
                    lambda changed, b_id=block.id, a=axis:
                        self._linear_component_edited(b_id, a, changed)
                )

            # Label is a separate, non-interactive widget so it cannot steal
            # keyboard focus from the field or intercept block dragging.
            with ui.Placer(
                offset_x=ui.Pixel(base * scale), offset_y=top,
                width=ui.Pixel(label_width * scale), height=field_height,
                stable_size=True,
            ):
                ui.Label(
                    axis.upper(), width=ui.Pixel(label_width * scale),
                    height=field_height, alignment=ui.Alignment.CENTER,
                    opaque_for_mouse_events=False,
                    style={"color": 0xFFFFFFFF,
                           "font_size": max(9, int(12 * scale))},
                )

            field_x = base + label_width + label_to_field_gap
            width = ui.Pixel(field_width * scale)
            with ui.Placer(
                offset_x=ui.Pixel(field_x * scale), offset_y=top,
                width=width, height=field_height, stable_size=True,
            ):
                field = ui.StringField(
                    model=model, width=width, height=field_height, enabled=True,
                    style={"color": 0xFF1A1A1A, "font_size": max(8, int(11 * scale)), "margin": 0},
                    tooltip=(f"{axis.upper()} absolute Zielkoordinate (m)" if block.type == "move_linear" else f"{axis.upper()} relative Verschiebung (m)"),
                    mouse_pressed_fn=lambda x, y, button, mods, b_id=block.id, a=axis:
                        self._linear_field_pressed(b_id, a, button),
                )
                self._linear_fields[key] = field

    def _linear_field_pressed(self, block_id: str, axis: str, button: Any) -> None:
        if self._runner.is_running or block_id not in self._program.blocks:
            return
        try:
            if int(button) not in (0, 1):
                return
        except (ValueError, TypeError):
            pass
        self._focus_now(self._linear_fields.get((block_id, axis)))
        asyncio.ensure_future(self._focus_linear_next_frame(block_id, axis))

    async def _focus_linear_next_frame(self, block_id: str, axis: str) -> None:
        import omni.kit.app
        await omni.kit.app.get_app().next_update_async()
        field = self._linear_fields.get((block_id, axis))
        if field is not None:
            try:
                field.focus_keyboard(True)
            except Exception as error:
                self._set_status(f"MOVE LINEAR: Fokusfehler: {error}")

    def _linear_component_edited(self, block_id: str, axis: str, model: Any) -> None:
        block = self._program.blocks.get(block_id)
        if block is None or block.type not in ("move_linear", "move_relative") or self._runner.is_running:
            return
        try:
            number = float(model.get_value_as_string().strip().replace(",", "."))
            if not math.isfinite(number):
                raise ValueError("non-finite")
            updated = dict(block.value)
            updated[axis] = number
            updated["coordinate_mode"] = "absolute" if block.type == "move_linear" else "relative"
            block.value = validate_linear_parameters(updated)
            self._set_status(f"{block.type.upper()}: {axis.upper()} = {number:g} m")
        except (ValueError, TypeError, OverflowError) as error:
            self._set_status(f"MOVE LINEAR: Ungueltige {axis.upper()}-Eingabe: {error}")

    # ------------------------------------------------------------------
    # MOVE JOINT: dropdown + direct degree input (no additional popup)
    # ------------------------------------------------------------------
    def _robot_for_joint_block(self, block_id: str) -> dict | None:
        """Use the most recent INIT ROBOT preceding MOVE JOINT in START's chain."""
        selected = None
        for item in self._program.execution_blocks():
            if item.id == block_id:
                return selected
            if item.type == "initialize_robot":
                try:
                    selected = validate_initialize_parameters(item.value)
                except (TypeError, ValueError):
                    selected = None
        return None

    def _joint_names_for_block(self, block_id: str) -> list[str]:
        """Detect rotational joints in the selected robot's actual USD stage."""
        selection = self._robot_for_joint_block(block_id)
        if selection is None:
            return []
        return self._cached_joint_names(selection["robot_path"])

    def _build_move_joint_inline(self, block: ProgramBlock, left: float, top: float) -> None:
        """Two separate fields on the MOVE JOINT block: joint menu and degrees."""
        scale = self._zoom
        field_height = ui.Pixel(WIDGET_HEIGHT * scale)
        dropdown_width = ui.Pixel(100 * scale)
        drop = {"accept_drop_fn": self._accept_drop, "drop_fn": self._drop_on_canvas}
        on_press = lambda x, y, button, m, b_id=block.id: self._joint_field_pressed(b_id, button)
        joint_name = str(block.value.get("joint_name", "")) or "Select joint"
        with ui.Placer(offset_x=ui.Pixel(left), offset_y=ui.Pixel(top),
                       width=dropdown_width, height=field_height, stable_size=True):
            with ui.ZStack(width=dropdown_width, height=field_height, style={"margin": 0}, **drop):
                ui.Rectangle(width=dropdown_width, height=field_height,
                             mouse_pressed_fn=on_press, **drop,
                             style={"background_color": 0xFFFFFFFF, "border_radius": 5})
                with ui.HStack(width=dropdown_width, height=field_height):
                    ui.Spacer(width=ui.Pixel(5 * scale))
                    self._joint_labels[block.id] = ui.Label(joint_name, height=field_height,
                             alignment=ui.Alignment.LEFT_CENTER, elided_text=True,
                             mouse_pressed_fn=on_press, **drop,
                             tooltip=f"Joint: {joint_name}\nClick to select a joint",
                             style={"color": 0xFF202020, "font_size": max(8, int(12 * scale))})
                    with ui.VStack(width=ui.Pixel(17 * scale), height=field_height):
                        ui.Spacer()
                        with ui.HStack(height=ui.Pixel(7 * scale)):
                            ui.Spacer()
                            ui.Triangle(width=ui.Pixel(8 * scale), height=ui.Pixel(6 * scale),
                                        alignment=ui.Alignment.CENTER_BOTTOM,
                                        mouse_pressed_fn=on_press,
                                        style={"background_color": 0xFF606060})
                            ui.Spacer()
                        ui.Spacer()

    def _build_joint_degree_field(self, block: ProgramBlock) -> None:
        """Editable text control, above draggable canvas surfaces and labels.

        A FloatField is mouse-draggable in omni.ui and may not enter text editing
        on a single click. A StringField provides normal click-to-type behavior.
        """
        scale = self._zoom
        # Compact layout: dropdown ends at x+202; angle occupies x+208..278;
        # delete X begins at x+300. All coordinates scale with canvas zoom.
        left = (block.x + 208) * scale
        top = (block.y + WIDGET_TOP) * scale
        width = ui.Pixel(70 * scale)
        height = ui.Pixel(WIDGET_HEIGHT * scale)

        model = self._joint_degree_models.get(block.id)
        if model is None:
            model = ui.SimpleStringModel(f"{float(block.value.get('target_degrees', 0.0)):g}")
            self._joint_degree_models[block.id] = model
            model.add_end_edit_fn(
                lambda changed, b_id=block.id: self._joint_degrees_edited(b_id, changed)
            )
        with ui.Placer(offset_x=ui.Pixel(left), offset_y=ui.Pixel(top),
                       width=width, height=height, stable_size=True):
            # Explicitly focus the field: click events within a drag-enabled
            # Scratch canvas otherwise may not give StringField keyboard focus.
            field = ui.StringField(
                model=model, width=width, height=height, enabled=True,
                style={"color": 0xFF1A1A1A, "font_size": max(8, int(12 * scale)), "margin": 0},
                tooltip="Zielwinkel in Grad: anklicken und Zahl eingeben",
                mouse_pressed_fn=lambda x, y, button, modifiers, b_id=block.id:
                    self._joint_degree_pressed(b_id, button),
            )
            self._joint_degree_fields[block.id] = field

    def _joint_degree_pressed(self, block_id: str, button: Any) -> None:
        """Give the inline angle field keyboard focus after its mouse event."""
        if self._runner.is_running or block_id not in self._program.blocks:
            return
        try:
            if int(button) not in (0, 1):
                return
        except (TypeError, ValueError):
            pass
        self._set_status("MOVE JOINT: Grad-Feld aktiviert - Winkel eintippen")
        self._focus_now(self._joint_degree_fields.get(block_id))
        asyncio.ensure_future(self._focus_joint_degree_next_frame(block_id))

    async def _focus_joint_degree_next_frame(self, block_id: str) -> None:
        import omni.kit.app
        await omni.kit.app.get_app().next_update_async()
        field = self._joint_degree_fields.get(block_id)
        if field is not None and block_id in self._program.blocks:
            try:
                field.focus_keyboard(True)
            except Exception as error:
                self._set_status(f"MOVE JOINT: Fokusfehler: {error}")

    def _joint_degrees_edited(self, block_id: str, model: Any) -> None:
        block = self._program.blocks.get(block_id)
        if block is None or block.type != "move_joint" or self._runner.is_running:
            return
        raw = model.get_value_as_string().strip().replace(",", ".")
        try:
            angle = float(raw)
            if not math.isfinite(angle):
                raise ValueError("Angle must be finite")
        except (ValueError, TypeError):
            self._set_status("MOVE JOINT: Bitte einen gültigen Winkel eingeben")
            return
        block.value = {**block.value, "target_degrees": angle}
        self._set_status(f"MOVE JOINT: Winkel auf {angle:g}° gesetzt")

    def _joint_field_pressed(self, block_id: str, button: Any) -> None:
        if self._runner.is_running or block_id not in self._program.blocks:
            return
        try:
            if int(button) not in (0, 1):
                return
        except (TypeError, ValueError):
            pass
        asyncio.ensure_future(self._show_joint_menu_next_frame(block_id))

    async def _show_joint_menu_next_frame(self, block_id: str) -> None:
        import omni.kit.app
        await omni.kit.app.get_app().next_update_async()
        block = self._program.blocks.get(block_id)
        if block is None or self._runner.is_running:
            return
        selection = self._robot_for_joint_block(block_id)
        if selection is not None:
            # Opening the menu is a deliberate user action: rescan this robot
            # so the list reflects the current stage.
            self._joint_cache.pop(selection["robot_path"], None)
        self._joint_menu = ui.Menu("MOVE JOINT – Choose joint")
        with self._joint_menu:
            if selection is None:
                ui.MenuItem("Connect INIT ROBOT before MOVE JOINT", enabled=False)
            else:
                try:
                    names = self._joint_names_for_block(block_id)
                except Exception as error:
                    self._set_status(f"Joint scan failed: {error}")
                    names = []
                if not names:
                    ui.MenuItem("No rotational joints found in selected robot", enabled=False)
                else:
                    for name in names:
                        ui.MenuItem(
                            name, checkable=True,
                            checked=(name == block.value.get("joint_name")),
                            triggered_fn=lambda b_id=block_id, j=name: self._joint_selected(b_id, j),
                        )
        self._joint_menu.show()

    def _joint_selected(self, block_id: str, joint_name: str) -> None:
        """Apply the choice now and change only the label text (no rebuild)."""
        block = self._program.blocks.get(block_id)
        label = self._joint_labels.get(block_id)
        if block is None or block.type not in ("move_joint", "wave") or self._runner.is_running:
            return
        if label is None:  # no label to update: fall back to the old rebuild path
            asyncio.ensure_future(self._apply_joint_selection_next_frame(block_id, joint_name))
            return
        if joint_name not in self._joint_names_for_block(block_id):
            self._set_status("Selected joint is no longer available in the stage")
            return
        block.value = {**block.value, "joint_name": joint_name}
        label.text = joint_name
        label.tooltip = f"Joint: {joint_name}\nClick to select a joint"
        self._set_status(f"MOVE JOINT: {joint_name} selected")

    async def _apply_joint_selection_next_frame(self, block_id: str, joint_name: str) -> None:
        import omni.kit.app
        await omni.kit.app.get_app().next_update_async()
        block = self._program.blocks.get(block_id)
        if block is None or self._runner.is_running or block.type not in ("move_joint", "wave"):
            return
        if joint_name not in self._joint_names_for_block(block_id):
            self._set_status("Selected joint is no longer available in the stage")
            return
        block.value = {**block.value, "joint_name": joint_name}
        self._refresh_canvas()
        self._set_status(f"MOVE JOINT: {joint_name} selected")

    # ------------------------------------------------------------------
    # INIT ROBOT: robot menu directly on the block (like the Joint Inspector)
    # ------------------------------------------------------------------
    def _scan_robots(self) -> list[str]:
        try:
            from .robot_scan import find_articulation_roots
            return find_articulation_roots()
        except Exception as error:  # no stage / pxr not ready
            print(f"[Scratch Blocks] Robot scan failed: {error}")
            return []

    @staticmethod
    def _guess_profile(robot_path: str, current: str) -> str:
        """Choose the profile automatically - the user only picks the robot.

        1. The profile whose arm joints all exist on the robot in the stage.
        2. Otherwise a profile whose name appears in the robot path.
        3. Otherwise use the generic robot profile, never the robot's prim name.
        """
        import json
        from .robot_profiles import PROFILE_FILE
        from .robot_scan import joint_names_of
        try:
            with open(PROFILE_FILE, encoding="utf-8") as stream:
                profiles = json.load(stream).get("profiles", {})
        except (OSError, ValueError):
            return "generic"
        joints = set(joint_names_of(robot_path))
        best, best_count = None, 0
        for name, data in profiles.items():
            arm = list((data or {}).get("arm_joint_names", []))
            gripper = list(((data or {}).get("gripper") or {}).get("joint_names", []))
            # Arm + mounted gripper are one robot: the profile matches only if
            # its gripper joints are in the stage too; more matched joints wins.
            if arm and set(arm) <= joints and set(gripper) <= joints:
                count = len(arm) + len(gripper)
                if count > best_count:
                    best, best_count = name, count
        if best:
            return best
        robot_name = robot_path.lower().rsplit("/", 1)[-1]
        for name in profiles:
            if name.lower() in robot_name:
                return name
        if ("panda" in robot_name or "franka" in robot_name) and "franka" in profiles:
            return "franka"
        return "generic" if "generic" in profiles else current

    def _build_init_robot_selector(self, block: ProgramBlock, left: float, top: float,
                                width: Any, height: Any) -> None:
        """Robot selector drawn directly on the INIT ROBOT block.

        A ui.ComboBox popup does not open reliably on the drag-enabled canvas,
        so the field is a label that opens a ui.Menu at the mouse position.
        The stage is scanned at click time, so the list is always current.
        """
        value = validate_initialize_parameters(block.value)
        on_press = lambda x, y, button, m, b_id=block.id: self._init_field_pressed(b_id, button)
        path = value["robot_path"]
        label_text = path.rstrip("/").rsplit("/", 1)[-1] or path  # /World/lara8 -> lara8
        self._build_menu_field(left, top, width, height, label_text,
                               f"{path}\nClick to choose the robot", on_press)

    def _build_init_gripper_selector(self, block: ProgramBlock, left: float, top: float,
                                     width: Any, height: Any) -> None:
        """Gripper selector on the INIT GRIPPER block (same look as INIT ROBOT)."""
        from .model import validate_gripper_init_parameters
        value = validate_gripper_init_parameters(block.value, require_selection=False)
        on_press = lambda x, y, button, m, b_id=block.id: self._init_gripper_field_pressed(b_id, button)
        path = value["gripper_path"]
        label_text = (path.rstrip("/").rsplit("/", 1)[-1] or path) if path else "choose gripper"
        tooltip = (f"{path} (profile {value['profile']})\nClick to choose the gripper" if path
                   else "Click to choose the gripper to mount on the INIT ROBOT arm")
        self._build_menu_field(left, top, width, height, label_text, tooltip, on_press)

    def _build_menu_field(self, left: float, top: float, width: Any, height: Any,
                          label_text: str, tooltip: str, on_press: Callable) -> None:
        """White field with a down arrow that opens a menu (INIT ROBOT / INIT GRIPPER)."""
        scale = self._zoom
        font = max(8, int(13 * scale))
        # The field must also accept drops, otherwise a block released over it
        # (e.g. when snapping another block under INIT ROBOT) is ignored.
        drop = {"accept_drop_fn": self._accept_drop, "drop_fn": self._drop_on_canvas}
        with ui.Placer(offset_x=ui.Pixel(left), offset_y=ui.Pixel(top),
                       width=width, height=height, stable_size=True):
            with ui.ZStack(width=width, height=height, style={"margin": 0}, **drop):  # margin 0: the window style {'margin': 7} would shrink these widgets
                ui.Rectangle(width=width, height=height, mouse_pressed_fn=on_press, **drop,
                             style={"background_color": 0xFFFFFFFF, "border_radius": 10})
                with ui.HStack(width=width, height=height):
                    ui.Spacer(width=ui.Pixel(8 * scale))
                    ui.Label(label_text, height=height,
                             alignment=ui.Alignment.LEFT_CENTER, elided_text=True,
                             mouse_pressed_fn=on_press, **drop,
                             tooltip=tooltip,
                             style={"color": 0xFF202020, "font_size": font})
                    # Drawn arrow: the Isaac Sim UI font has no "\u25BC" glyph (showed "?").
                    with ui.VStack(width=ui.Pixel(24 * scale), height=height):
                        ui.Spacer()
                        with ui.HStack(height=ui.Pixel(8 * scale)):
                            ui.Spacer()
                            ui.Triangle(width=ui.Pixel(10 * scale), height=ui.Pixel(8 * scale),
                                        alignment=ui.Alignment.CENTER_BOTTOM,
                                        mouse_pressed_fn=on_press,
                                        style={"background_color": 0xFF606060})
                            ui.Spacer()
                        ui.Spacer()

    def _init_field_pressed(self, block_id: str, button: Any) -> None:
        if self._runner.is_running or block_id not in self._program.blocks:
            return
        try:
            if int(button) not in (0, 1):
                return
        except (TypeError, ValueError):
            pass
        # Open the menu one frame later, after the mouse event is finished.
        asyncio.ensure_future(self._show_robot_menu_next_frame(block_id))

    async def _show_robot_menu_next_frame(self, block_id: str) -> None:
        import omni.kit.app
        await omni.kit.app.get_app().next_update_async()
        block = self._program.blocks.get(block_id)
        if block is None:
            return
        value = validate_initialize_parameters(block.value)
        robots = self._scan_robots()
        self._robot_menu = ui.Menu("INIT ROBOT")
        with self._robot_menu:
            if robots:
                for path in robots:
                    ui.MenuItem(
                        path, checkable=True, checked=(path == value["robot_path"]),
                        triggered_fn=lambda b_id=block_id, p=path: self._init_robot_selected(b_id, p),
                    )
            else:
                ui.MenuItem("No articulation detected: check loaded USD stage", enabled=False)
        self._robot_menu.show()

    def _init_robot_selected(self, block_id: str, selection: str) -> None:
        # Never rebuild the canvas inside a widget callback -> defer one frame.
        asyncio.ensure_future(self._apply_init_robot_next_frame(block_id, selection))

    async def _apply_init_robot_next_frame(self, block_id: str, selection: str) -> None:
        import omni.kit.app
        await omni.kit.app.get_app().next_update_async()
        block = self._program.blocks.get(block_id)
        if block is None or self._runner.is_running:
            self._refresh_canvas()
            return
        if not selection.startswith("/"):
            return
        old = validate_initialize_parameters(block.value)
        profile = self._guess_profile(selection, old["profile"])
        block.value = {"robot_path": selection, "profile": profile}
        self._invalidate_scene_cache()  # New robot: rescan joints and profile.
        from .model import retarget_joint_names
        changed, missing = retarget_joint_names(self._program, self._cached_joint_names(selection))
        self._all_joint_models.clear()  # Rebuild input fields for selected robot.
        self._joint_degree_models.clear()
        self._remount_grippers()  # INIT GRIPPER follows the newly chosen arm.
        self._refresh_canvas()
        message = f"INIT ROBOT: {selection} (profile {profile})"
        if changed:
            message += f" - {changed} joint block(s) matched to this robot by joint number"
        if missing:
            message += f" - NOT on this robot: {', '.join(missing)}"
        self._set_status(message)

    # ------------------------------------------------------------------
    # INIT GRIPPER: choose a gripper in the stage; it is mounted on the
    # flange of the arm from the INIT ROBOT block above (simulation stopped).
    # ------------------------------------------------------------------
    def _init_gripper_field_pressed(self, block_id: str, button: Any) -> None:
        if self._runner.is_running or block_id not in self._program.blocks:
            return
        try:
            if int(button) not in (0, 1):
                return
        except (TypeError, ValueError):
            pass
        asyncio.ensure_future(self._show_gripper_init_menu_next_frame(block_id))

    @staticmethod
    def _scan_grippers() -> list[str]:
        try:
            import omni.usd
            from .gripper_mount import find_grippers
            from .robot_profiles import gripper_profiles
            stage = omni.usd.get_context().get_stage()
            return find_grippers(stage, gripper_profiles()) if stage is not None else []
        except Exception as error:
            print(f"[Scratch Blocks] Gripper scan failed: {error}")
            return []

    async def _show_gripper_init_menu_next_frame(self, block_id: str) -> None:
        import omni.kit.app
        await omni.kit.app.get_app().next_update_async()
        block = self._program.blocks.get(block_id)
        if block is None:
            return
        from .model import validate_gripper_init_parameters
        current = validate_gripper_init_parameters(block.value, require_selection=False)["gripper_path"]
        grippers = self._scan_grippers()
        self._gripper_init_menu = ui.Menu("INIT GRIPPER")
        with self._gripper_init_menu:
            if grippers:
                for path in grippers:
                    ui.MenuItem(
                        path, checkable=True, checked=(path == current),
                        triggered_fn=lambda b_id=block_id, p=path: self._init_gripper_selected(b_id, p),
                    )
            else:
                ui.MenuItem("No gripper found: drag a gripper USD into the stage", enabled=False)
            if current:
                ui.Separator()
                ui.MenuItem("Unmount gripper", triggered_fn=lambda b_id=block_id: self._init_gripper_selected(b_id, ""))
        self._gripper_init_menu.show()

    def _init_gripper_selected(self, block_id: str, selection: str) -> None:
        asyncio.ensure_future(self._apply_init_gripper_next_frame(block_id, selection))

    async def _apply_init_gripper_next_frame(self, block_id: str, selection: str) -> None:
        import omni.kit.app
        await omni.kit.app.get_app().next_update_async()
        block = self._program.blocks.get(block_id)
        if block is None or self._runner.is_running:
            self._refresh_canvas()
            return
        try:
            summary = self._mount_gripper_block(block, selection)
        except Exception as error:
            self._set_status(f"INIT GRIPPER: {error}")
            carb.log_error(f"[Scratch Blocks] INIT GRIPPER failed: {error}")
            return
        self._invalidate_scene_cache()
        self._refresh_canvas()
        self._set_status(summary)
        print(f"[Scratch Blocks] {summary}")

    def _mount_gripper_block(self, block: ProgramBlock, selection: str) -> str:
        """Mount (or unmount) the selected gripper on this block's INIT ROBOT arm."""
        import omni.timeline
        import omni.usd
        from . import gripper_mount as gm
        from .model import validate_gripper_init_parameters
        from .robot_profiles import gripper_profiles
        if omni.timeline.get_timeline_interface().is_playing():
            raise RuntimeError("press Stop first - a gripper can only be mounted while the simulation is stopped")
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("no USD stage loaded")
        old = validate_gripper_init_parameters(block.value, require_selection=False)
        if not selection:
            if old["gripper_path"]:
                gm.unmount(stage, old["gripper_path"])
            block.value = {"gripper_path": "", "profile": ""}
            return f"INIT GRIPPER: {old['gripper_path'] or 'gripper'} unmounted"
        arm = self._robot_for_joint_block(block.id)
        if arm is None:
            raise RuntimeError("connect INIT GRIPPER under an INIT ROBOT block first")
        profiles = gripper_profiles()
        profile = gm.match_gripper_profile(stage, selection, profiles)
        if profile is None:
            raise RuntimeError(f"no gripper profile matches {selection}: add it to the "
                               "'grippers' section of robot_profiles.json")
        if old["gripper_path"] and old["gripper_path"] != selection:
            gm.unmount(stage, old["gripper_path"])
        summary = gm.mount(stage, arm["robot_path"], selection, self._cached_profile(arm["profile"]))
        block.value = {"gripper_path": selection, "profile": profile}
        flange = summary["flange"].rsplit("/", 1)[-1]
        return (f"INIT GRIPPER: {selection} mounted on {arm['robot_path']} at {flange} "
                f"(profile {profile}) - press Play to use it")

    def _remount_grippers(self) -> None:
        """After INIT ROBOT changes arm: move INIT GRIPPER grippers to the new arm."""
        from .model import validate_gripper_init_parameters
        for item in self._program.execution_blocks():
            if item.type != "initialize_gripper":
                continue
            path = validate_gripper_init_parameters(item.value, require_selection=False)["gripper_path"]
            if not path:
                continue
            try:
                self._mount_gripper_block(item, path)
            except Exception as error:
                self._set_status(f"INIT GRIPPER: {error}")
                carb.log_warn(f"[Scratch Blocks] gripper remount skipped: {error}")

    def _parameter_pressed(self, block_id: str, button: Any) -> None:
        if self._runner.is_running or block_id not in self._program.blocks:
            return
        try:
            if int(button) not in (0, 1):
                return
        except (TypeError, ValueError):
            pass
        asyncio.ensure_future(self._open_parameter_next_frame(block_id))

    async def _open_parameter_next_frame(self, block_id: str) -> None:
        import omni.kit.app
        await omni.kit.app.get_app().next_update_async()
        block = self._program.blocks.get(block_id)
        if block is None:
            return
        if self._parameter_window is not None:
            self._parameter_window.visible = False
        if block.type in ("set_speed", "repeat", "end_repeat", "stop_program", "initialize_gripper"):
            return  # Edited inline (INIT GRIPPER: menu on the block), or has no parameters.
        if block.type in ("move_joint", "wave"):
            return  # Joint and numbers are edited directly inside the block.
        if block.type == "move_all_joints":
            return  # All joint angles are edited directly in the block.
        if block.type == "gripper":
            return  # Open/Close is selected directly in the block.
        if block.type in ("move_linear", "move_relative"):
            self._open_linear_parameter(block)
            return
        self._parameter_window = ui.Window("Edit Block Parameter", width=420, height=265 if block.type == "move_joint" else 155)
        with self._parameter_window.frame:
            with ui.VStack(spacing=10, style={"margin": 12}):
                ui.Label("Edit " + block.type + " parameter:", height=22)
                if block.type == "move_joint":
                    ui.Label("Joint name", height=18)
                    joint_model = ui.SimpleStringModel(str(block.value["joint_name"]))
                    ui.StringField(model=joint_model, height=28)
                    ui.Label("Target angle (degrees)", height=18)
                    angle_model = ui.SimpleFloatModel(float(block.value["target_degrees"]))
                    ui.FloatField(model=angle_model, height=28)
                    with ui.HStack(spacing=8, height=30):
                        ui.Button("Apply", clicked_fn=lambda b_id=block_id, jm=joint_model, am=angle_model: self._apply_joint_parameter(b_id, jm, am))
                        ui.Button("Cancel", clicked_fn=self._close_parameter_window)
                else:
                    if block.type in ("wait", "set_speed"):
                        model = ui.SimpleFloatModel(float(block.value))
                        ui.FloatField(model=model, height=28)
                    else:
                        model = ui.SimpleStringModel(str(block.value))
                        ui.StringField(model=model, height=28)
                    with ui.HStack(spacing=8, height=30):
                        ui.Button("Apply", clicked_fn=lambda b_id=block_id, m=model: self._apply_parameter(b_id, m))
                        ui.Button("Cancel", clicked_fn=self._close_parameter_window)

    def _open_linear_parameter(self, block: ProgramBlock) -> None:
        self._parameter_window = ui.Window("Edit Block Parameter", width=560, height=610)
        models = {}
        fields = (
            ("steps", "Number of path sections (integer)"),
            ("duration", "Duration per section (s); total = sections x duration"),
            ("robot_description_path", "Robot description YAML path"),
            ("urdf_path", "Robot URDF path"),
            ("end_effector_frame", "End-effector frame"),
        )
        with self._parameter_window.frame:
            with ui.VStack(spacing=6, style={"margin": 12}):
                ui.Label(block.type.upper() + " - Advanced settings", height=24)
                ui.Label("X, Y and Z remain editable directly in the block.", height=24)
                ui.Label("Leave configuration fields empty to use the INIT profile.", height=24)
                for key, label in fields:
                    ui.Label(label, height=18)
                    model = ui.SimpleStringModel(str(block.value.get(key, "")))
                    models[key] = model
                    ui.StringField(model=model, height=28)
                with ui.HStack(spacing=8, height=30):
                    ui.Button("Apply", clicked_fn=lambda b_id=block.id, ms=models: self._apply_linear_parameter(b_id, ms))
                    ui.Button("Cancel", clicked_fn=self._close_parameter_window)

    def _apply_linear_parameter(self, block_id: str, models: dict) -> None:
        block = self._program.blocks.get(block_id)
        if block is None or block.type not in ("move_linear", "move_relative") or self._runner.is_running:
            self._close_parameter_window()
            return
        try:
            # Merge the advanced fields into the current block parameters.
            # Never overwrite the inline X/Y/Z coordinates with stale popup data.
            updated = dict(block.value)
            updated.update({key: model.get_value_as_string() for key, model in models.items()})
            updated["coordinate_mode"] = "absolute" if block.type == "move_linear" else "relative"
            value = validate_linear_parameters(updated)
        except (TypeError, ValueError, OverflowError) as error:
            self._set_status(f"MOVE LINEAR: {error}")
            return
        block.value = value
        for axis in ("x", "y", "z"):
            self._linear_models.pop((block_id, axis), None)
        self._close_parameter_window()
        self._refresh_canvas()
        self._set_status("MOVE LINEAR parameters updated")

    def _apply_all_parameter(self, block_id, model):
        try:
            text = model.get_value_as_string().strip()
            parse_joint_targets(text)
        except (TypeError, ValueError) as error:
            self._set_status(f"MOVE ALL: {error}")
            return
        block = self._program.blocks.get(block_id)
        if block is not None and block.type == "move_all_joints" and not self._runner.is_running:
            block.value = {"targets": text}
        self._close_parameter_window()
        self._refresh_canvas()
        self._set_status("MOVE ALL parameters updated")

    def _apply_joint_parameter(self, block_id: str, joint_model: Any, angle_model: Any) -> None:
        block = self._program.blocks.get(block_id)
        if block is None or block.type != "move_joint":
            self._close_parameter_window()
            return
        joint_name = joint_model.get_value_as_string().strip()
        angle = float(angle_model.get_value_as_float())
        if not joint_name or not math.isfinite(angle):
            self._set_status("Invalid joint name or target angle")
            return
        block.value = {"joint_name": joint_name, "target_degrees": angle}
        self._close_parameter_window()
        self._refresh_canvas()
        self._set_status("MOVE JOINT parameters updated")

    def _apply_parameter(self, block_id: str, model: Any) -> None:
        block = self._program.blocks.get(block_id)
        if block is None:
            self._close_parameter_window()
            return
        if block.type == "gripper":
            value = model.get_value_as_string().strip().lower()
            if value not in ("open", "close"):
                self._set_status("GRIPPER: enter open or close")
                return
            block.value = value
        elif block.type == "set_speed":
            try:
                block.value = validate_speed(model.get_value_as_float())
            except (ValueError, TypeError) as error:
                self._set_status(str(error))
                return
        elif block.type == "wait":
            value = model.get_value_as_float()
            if not math.isfinite(value) or value < 0:
                self._set_status("Invalid wait duration")
                return
            block.value = value
        else:
            block.value = model.get_value_as_string()
        self._close_parameter_window()
        self._refresh_canvas()
        self._set_status("Parameter updated")

    def _close_parameter_window(self) -> None:
        if self._parameter_window is not None:
            self._parameter_window.visible = False
            self._parameter_window = None

    def _build_delete_button(self, block: ProgramBlock) -> None:
        """Draw the delete control above the draggable block surface.

        Use a mouse press callback rather than a nested ui.Button: the block
        surface accepts drag/drop events, which can interfere with Button clicks.
        """
        size_x = ui.Pixel(30 * self._zoom)
        size_y = ui.Pixel(WIDGET_HEIGHT * self._zoom)
        on_press = lambda x, y, button, modifiers, block_id=block.id: self._delete_x_pressed(block_id, button)
        with ui.Placer(
            offset_x=ui.Pixel((block.x + self._width_for_block(block) - 60) * self._zoom),
            offset_y=ui.Pixel((block.y + WIDGET_TOP) * self._zoom),
            width=size_x, height=size_y,
            stable_size=True,
        ):
            with ui.ZStack(width=size_x, height=size_y, style={"margin": 0},
                           accept_drop_fn=self._accept_drop, drop_fn=self._drop_on_canvas):
                ui.Rectangle(
                    width=size_x,
                    height=size_y,
                    accept_drop_fn=self._accept_drop,
                    drop_fn=self._drop_on_canvas,
                    style={"background_color": 0xFFB63333, "border_radius": 5},
                )
                ui.Label(
                    "X",
                    width=size_x,
                    height=size_y,
                    alignment=ui.Alignment.CENTER,
                    mouse_pressed_fn=on_press,
                    style={"color": 0xFFFFFFFF, "font_size": max(10, int(15 * self._zoom))},
                )

    def _delete_x_pressed(self, block_id: str, button: Any) -> None:
        """Handle a primary mouse click on an individual block's X."""
        # In omni.ui the primary button may be represented by 0 or 1,
        # depending on the event bindings. Filter only known non-primary buttons.
        try:
            if int(button) not in (0, 1):
                return
        except (TypeError, ValueError):
            pass
        if self._delete_pending or self._runner.is_running:
            return
        if block_id not in self._program.blocks:
            return

        # A mouse press can reach several nested/overlapping widgets.
        # Never rebuild the canvas from inside that mouse callback: doing so
        # can allow the SAME press to hit newly positioned X controls.
        self._delete_pending = True
        asyncio.ensure_future(self._delete_after_mouse_press(block_id))

    async def _delete_after_mouse_press(self, block_id: str) -> None:
        try:
            # Let the current UI event finish before removing/reflowing a block.
            import omni.kit.app
            app = omni.kit.app.get_app()
            await app.next_update_async()
            if block_id in self._program.blocks and not self._runner.is_running:
                self._remove(block_id)
            # Do not admit another deletion until the rebuilt UI gets a frame.
            await app.next_update_async()
        except Exception as error:
            self._set_status(f"Delete failed: {error}")
        finally:
            self._delete_pending = False

    def _accept_drop(self, mime_data: str) -> bool:
        return isinstance(mime_data, str) and mime_data.startswith(DRAG_PREFIX)

    def _parse_drop(self, event: Any) -> tuple[str, str]:
        mime_data = event if isinstance(event, str) else getattr(event, "mime_data", "")
        if not self._accept_drop(mime_data):
            raise ValueError("Unsupported drag payload")
        return tuple(mime_data[len(DRAG_PREFIX) :].split(":", 1))  # type: ignore[return-value]

    def _materialize_drop(self, action: str, value: str, x: float, y: float) -> ProgramBlock:
        if action == "new" and value in SUPPORTED_BLOCKS:
            return self._program.add(value, x, y)
        if action == "move":
            block = self._program.get(value)
            self._program.move_stack(block.id, x, y)
            return block
        raise ValueError("Invalid block drag")

    def _template_payload(self, block_type: str) -> str:
        # The grab point was already stored by _remember_template_anchor.
        return f"{DRAG_PREFIX}new:{block_type}"

    def _drop_on_canvas(self, event: Any) -> None:
        """Place (and possibly snap) a dropped block.

        A block only connects when it lands directly under another block;
        otherwise it stays exactly where it was released.
        """
        if self._runner.is_running or self._drop_handled:
            return
        # Several overlapping widgets (canvas, START, blocks, fields) accept the
        # same drop. Handle it once; the flag is cleared on the next frame.
        self._drop_handled = True
        asyncio.ensure_future(self._clear_drop_flag_next_frame())
        try:
            action, value = self._parse_drop(event)
            x, y = self._canvas_position(event)
            self._drag_anchor_x = self._drag_anchor_y = 0.0
            self._press = None
            block = self._materialize_drop(action, value, x, y)
            if action == "new" and block.type == "repeat":
                # A new REPEAT box always comes with its closing arm.
                end = self._program.add("end_repeat", x, y + BLOCK_STEP)
                self._program.connect_after(block.id, end.id)
                block.x, block.y = x, y
            insertion = self._nearest_insertion(x, y, block.id)
            if insertion is not None:
                mode, target_id = insertion
                if mode == "before":
                    self._program.connect_before(target_id, block.id)
                else:
                    self._program.connect_after(target_id, block.id)
                self._set_status(f"Connected {mode} {self._block_name(target_id)}")
            else:
                self._set_status(f"Placed at ({x:.0f}, {y:.0f})")
            self._refresh_canvas()
        except Exception as error:
            self._set_status(f"Drop failed: {error}")
            carb.log_warn(f"[Scratch Blocks] Drop failed: {error}")

    def _block_name(self, block_id: str) -> str:
        if block_id == "START":
            return "START"
        block = self._program.blocks.get(block_id)
        return block.type.upper() if block else block_id[:8]

    async def _clear_drop_flag_next_frame(self) -> None:
        import omni.kit.app
        await omni.kit.app.get_app().next_update_async()
        self._drop_handled = False

    # -- mouse position -----------------------------------------------------
    # The drop event's x/y is not reliable in Isaac Sim 6.2 (blocks landed
    # shifted or in the canvas corner). Instead the drag is measured with the
    # real mouse: position at the press (known exactly in canvas coordinates)
    # plus how far the mouse moved until the release.

    @staticmethod
    def _carb_mouse_pixels() -> tuple[float, float] | None:
        """Raw mouse position from carb (pixels in the app window)."""
        try:
            import omni.appwindow
            mouse = omni.appwindow.get_default_app_window().get_mouse()
            px, py = carb.input.acquire_input_interface().get_mouse_coords_pixel(mouse)
            return float(px), float(py)
        except Exception:
            return None

    def _pixels_per_ui_unit(self) -> float:
        try:
            return float(ui.Workspace.get_dpi_scale()) or 1.0
        except Exception:
            return 1.0

    def _remember_press(self, ui_x: float, ui_y: float) -> None:
        """Store the press point in omni.ui coordinates and in carb pixels."""
        raw = self._carb_mouse_pixels()
        self._press = None if raw is None else (float(ui_x), float(ui_y), raw[0], raw[1])

    def _release_ui_position(self, event: Any) -> tuple[float, float]:
        """Release point in omni.ui screen coordinates."""
        raw = self._carb_mouse_pixels()
        if self._press is not None and raw is not None:
            press_x, press_y, raw_x0, raw_y0 = self._press
            scale = self._pixels_per_ui_unit()
            return (press_x + (raw[0] - raw_x0) / scale,
                    press_y + (raw[1] - raw_y0) / scale)
        # Fallback: the drop event itself.
        return float(getattr(event, "x", 0.0)), float(getattr(event, "y", 0.0))

    def _canvas_position(self, event: Any) -> tuple[float, float]:
        """Top-left corner of the dropped block in logical canvas coordinates.

        release point - canvas origin - grab offset, divided by the zoom.
        Scrolling is included because the canvas surface's screen position
        moves with the scroll bars.
        """
        release_x, release_y = self._release_ui_position(event)
        surface = self._canvas_surface
        origin_x = float(surface.screen_position_x) if surface is not None else 0.0
        origin_y = float(surface.screen_position_y) if surface is not None else 0.0
        x = (release_x - origin_x - self._drag_anchor_x) / self._zoom
        y = (release_y - origin_y - self._drag_anchor_y) / self._zoom
        print(f"[Scratch Blocks] drop: event=({getattr(event, 'x', None)}, {getattr(event, 'y', None)}) "
              f"release=({release_x:.0f}, {release_y:.0f}) origin=({origin_x:.0f}, {origin_y:.0f}) "
              f"anchor=({self._drag_anchor_x:.0f}, {self._drag_anchor_y:.0f}) -> ({x:.0f}, {y:.0f})")
        return (
            max(0.0, min(CANVAS_WIDTH - BLOCK_WIDTH, x)),
            max(0.0, min(CANVAS_HEIGHT - BLOCK_HEIGHT, y)),
        )

    def _build_grid(self) -> None:
        """Draw fixed logical cells so drop locations are visually unambiguous."""
        scale = self._zoom
        step = GRID_SIZE // 2   # 50 px squares, every 5th line darker (250 px)
        for index, x in enumerate(range(step, CANVAS_WIDTH, step), start=1):
            color = GRID_MAJOR if index % 5 == 0 else GRID_MINOR
            with ui.Placer(offset_x=ui.Pixel(x * scale), stable_size=True):
                ui.Rectangle(
                    width=ui.Pixel(1),
                    height=ui.Pixel(CANVAS_HEIGHT * scale),
                    opaque_for_mouse_events=False,
                    style={"background_color": color},
                )
        for index, y in enumerate(range(step, CANVAS_HEIGHT, step), start=1):
            color = GRID_MAJOR if index % 5 == 0 else GRID_MINOR
            with ui.Placer(offset_y=ui.Pixel(y * scale), stable_size=True):
                ui.Rectangle(
                    width=ui.Pixel(CANVAS_WIDTH * scale),
                    height=ui.Pixel(1),
                    opaque_for_mouse_events=False,
                    style={"background_color": color},
                )

    def _remember_template_anchor(self, x: float, y: float, block_type: str) -> None:
        """Remember WHERE on the library block the user grabbed it.

        The new canvas block is then placed so that the same spot is under the
        mouse when it is released, i.e. exactly where the user drops it.
        The library block is smaller than a canvas block, so the grab point is
        scaled to canvas size (screen pixels at the current zoom).
        """
        self._remember_press(x, y)
        template = self._templates.get(block_type)
        if template is None:
            self._drag_anchor_x = BLOCK_WIDTH * self._zoom / 2
            self._drag_anchor_y = BLOCK_HEIGHT * self._zoom / 2
            return
        rel_x = min(max(0.0, float(x) - float(template.screen_position_x)), TEMPLATE_WIDTH)
        rel_y = min(max(0.0, float(y) - float(template.screen_position_y)), TEMPLATE_HEIGHT)
        self._drag_anchor_x = rel_x * BLOCK_WIDTH * self._zoom / TEMPLATE_WIDTH
        self._drag_anchor_y = rel_y * BLOCK_HEIGHT * self._zoom / TEMPLATE_HEIGHT

    def _remember_block_anchor(self, x: float, y: float, block: ProgramBlock) -> None:
        self._remember_press(x, y)
        surface = self._canvas_surface
        if surface is None:
            self._drag_anchor_x = self._drag_anchor_y = 0.0
            return
        block_screen_x = float(surface.screen_position_x) + block.x * self._zoom
        block_screen_y = float(surface.screen_position_y) + block.y * self._zoom
        self._drag_anchor_x = max(0.0, float(x) - block_screen_x)
        self._drag_anchor_y = max(0.0, float(y) - block_screen_y)

    def _on_canvas_wheel(self, x: float, y: float, modifier: Any) -> None:
        """Ctrl + wheel zoom; ordinary wheel remains available for scrolling."""
        if not self._has_control_modifier(modifier) or y == 0:
            return
        self._change_zoom(ZOOM_STEP if y > 0 else -ZOOM_STEP)

    @staticmethod
    def _has_control_modifier(modifier: Any) -> bool:
        flags = getattr(carb.input, "KeyboardModifierFlags", None)
        if flags is not None:
            for name in ("CONTROL", "CTRL"):
                control = getattr(flags, name, None)
                if control is not None:
                    return bool(int(modifier) & int(control))
        # Carbonite's control modifier bit in older Kit Python bindings.
        return bool(int(modifier) & 2)

    def _change_zoom(self, delta: float) -> None:
        self._set_zoom(self._zoom + delta)

    def _set_zoom(self, value: float) -> None:
        self._zoom = max(ZOOM_MIN, min(ZOOM_MAX, round(value / ZOOM_STEP) * ZOOM_STEP))
        if self._zoom_label is not None:
            self._zoom_label.text = f"{int(self._zoom * 100)}%"
        self._refresh_canvas()
        self._set_status(f"Canvas zoom: {int(self._zoom * 100)}%")

    def _nearest_insertion(
        self, x: float, y: float, moving_id: str
    ) -> tuple[str, str] | None:
        """Snap when the moving block's TOP lands directly under a block.

        Candidates after any block insert at that point of the chain, moving
        the rest downward. Candidates before an unconnected block also work.
        All coordinates are logical canvas coordinates and zoom-independent.
        """
        moving = set(self._program.chain_ids(moving_id))
        candidates: list[tuple[str, str, float, float]] = [
            ("after", "START", START_X, START_Y + START_HEIGHT),
        ]
        for target in self._program.blocks.values():
            if target.id in moving:
                continue
            candidates.append(("after", target.id, target.x, target.y + BLOCK_STEP))
            # A free block with no parent may also receive a block above it.
            if target.parent_id is None:
                candidates.append(("before", target.id, target.x, target.y))

        winner: tuple[float, str, str] | None = None
        for mode, target_id, tx, ty in candidates:
            dx, offset = abs(x - tx), y - ty
            if dx > SNAP_X_TOLERANCE:
                continue
            if mode == "after" and not (-SNAP_ABOVE <= offset <= SNAP_BELOW):
                continue
            # "before": the dropped block's bottom touches the target's top.
            if mode == "before" and not (-BLOCK_STEP - SNAP_ABOVE <= offset <= -BLOCK_STEP + SNAP_ABOVE):
                continue
            dy = abs(offset) if mode == "after" else abs(offset + BLOCK_STEP)
            # Prefer after-block candidates in ties (natural Scratch insertion).
            score = dy + dx * 0.05 + (0.01 if mode == "before" else 0.0)
            if winner is None or score < winner[0]:
                winner = (score, mode, target_id)
        return None if winner is None else (winner[1], winner[2])

    def _remove(self, block_id: str) -> None:
        if not self._runner.is_running:
            block = self._program.blocks.get(block_id)
            if block is not None and block.type == "repeat":
                # Deleting the box keeps its contents in place of the C.
                end = self._program.matching_end(block_id)
                if end is not None:
                    self._program.remove(end.id)
            self._program.remove(block_id)
            self._speed_models.pop(block_id, None)
            self._speed_fields.pop(block_id, None)
            self._joint_degree_models.pop(block_id, None)
            self._joint_degree_fields.pop(block_id, None)
            for key in list(self._all_joint_models):
                if key[0] == block_id:
                    self._all_joint_models.pop(key, None)
                    self._all_joint_fields.pop(key, None)
            for axis in ("x", "y", "z"):
                self._linear_models.pop((block_id, axis), None)
                self._linear_fields.pop((block_id, axis), None)
            for key, _caption, _tip in WAVE_FIELDS:
                self._wave_models.pop((block_id, key), None)
                self._wave_fields.pop((block_id, key), None)
            self._set_status("Block deleted")
            self._refresh_canvas()

    def _run(self) -> None:
        blocks = self._validate_program(report_success=False)
        if blocks is None:
            return
        if self._real_busy:
            self._set_status("Real robot is busy")
            return
        self._prepare_validation(blocks)
        if not self._runner.start(blocks):
            self._set_status("Already running")
            return
        # Rebuild once so drag handles are locked while the program runs.
        self._refresh_canvas()

    # ------------------------------------------------------------------
    # Act on real robot:  Sync -> Run (simulation) -> VALIDATED -> Act
    # ------------------------------------------------------------------
    def _update_act_button(self) -> None:
        if self._act_button is not None:
            self._act_button.text = ("Act on real" if self._validation is not None
                                     else "Act on real (locked)")

    def _invalidate(self, reason: str | None = None) -> None:
        self._validation = None
        self._pending_validation = None
        self._update_act_button()
        if reason:
            self._set_status(reason)

    def _sync_from_real(self) -> None:
        if self._runner.is_running or self._real_busy:
            self._set_status("Wait until the current run has finished")
            return
        asyncio.ensure_future(self._sync_async())

    async def _sync_async(self) -> None:
        """Read the real arm, then put the simulated arm in exactly that pose."""
        from . import real_robot
        self._real_busy = True
        try:
            self._sim_arm = self._find_sim_arm()
            self._set_status("Sync: reading the real robot ...")
            if self._real_reader is None:
                self._real_reader = await asyncio.to_thread(real_robot.connect)
            angles = [float(a) for a in
                      await asyncio.to_thread(self._real_reader.get_current_joint_angles)]
            if len(angles) != len(self._sim_arm.dof_names):
                raise ValueError(f"Real robot has {len(angles)} joints, simulated arm "
                                 f"{len(self._sim_arm.dof_names)}: wrong robot model in the scene?")
            self._apply_sim_pose(angles)
            self._sync_pose = angles
            self._invalidate()
            degrees = [round(math.degrees(a), 1) for a in angles]
            gripper = self._sync_sim_gripper()
            self._set_status(f"Synced from real robot {degrees} deg{gripper} - now press Run to validate")
            print(f"[Scratch Blocks][real] synced pose (deg): {degrees}")
        except Exception as error:
            self._set_status(f"Sync failed: {error}")
            carb.log_error(f"[Scratch Blocks][real] sync failed: {error}")
        finally:
            self._real_busy = False

    def _find_sim_arm(self):
        """Isaac Sim handle for the robot named in the program's INIT ROBOT block."""
        import omni.timeline
        if not omni.timeline.get_timeline_interface().is_playing():
            raise RuntimeError("press Play first")
        init = next((b for b in self._program.execution_blocks()
                     if b.type == "initialize_robot"), None)
        if init is None:
            raise RuntimeError("the program needs an INIT ROBOT block")
        from .robot_init import initialize_robot
        from .robot_profiles import load_profile
        from .arm_view import arm_view_for_profile
        value = validate_initialize_parameters(init.value)
        robot = initialize_robot(value["robot_path"])
        # The real arm only knows its arm joints; hide mounted-gripper DOFs
        # (e.g. finger_stroke) from Sync, validation and the real-run mirror.
        return arm_view_for_profile(robot, load_profile(value["profile"]))

    # --- simulated gripper follows the REAL gripper ---------------------
    def _gripper_config(self) -> dict | None:
        """Gripper of this program: INIT GRIPPER's profile, else the arm profile's own."""
        from .model import validate_gripper_init_parameters
        from .robot_profiles import load_gripper_profile, load_profile
        blocks = self._program.execution_blocks()
        for block in blocks:
            if block.type == "initialize_gripper":
                value = validate_gripper_init_parameters(block.value, require_selection=False)
                if value["profile"]:
                    return load_gripper_profile(value["profile"])
        init = next((b for b in blocks if b.type == "initialize_robot"), None)
        if init is not None:
            return load_profile(validate_initialize_parameters(init.value)["profile"]).get("gripper")
        return None

    def _set_sim_gripper(self, action: str, teleport: bool) -> None:
        """Open/close the simulated gripper. teleport=True also sets the fingers
        there immediately (Sync); False only sends the drive target (live run)."""
        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction
        config = self._gripper_config()
        if not config or self._sim_arm is None:
            raise RuntimeError("no gripper configured in this program")
        articulation = getattr(self._sim_arm, "_art", self._sim_arm)  # full arm+gripper
        names = list(articulation.dof_names)
        missing = [n for n in config["joint_names"] if n not in names]
        if missing:
            raise RuntimeError(f"gripper joints {missing} not in the simulated robot (mounted before Play?)")
        indices = np.asarray([names.index(n) for n in config["joint_names"]], dtype=np.int32)
        target = np.asarray(config["open_positions" if action == "open" else "closed_positions"], dtype=float)
        if teleport:
            articulation.set_joint_positions(target, joint_indices=indices)
        articulation.apply_action(ArticulationAction(joint_positions=target, joint_indices=indices))

    def _mirror_real_gripper(self, action: str) -> None:
        """Runner callback: the real gripper was just commanded -> sim follows."""
        self._set_sim_gripper(action, teleport=False)

    def _sync_sim_gripper(self) -> str:
        """Part of Sync: open the sim gripper - every run starts with INIT GRIPPER
        opening the gripper (real and sim), so open is the known starting state."""
        try:
            if self._gripper_config() is None:
                return ""
            self._set_sim_gripper("open", teleport=True)
        except Exception as error:
            return f" | gripper not synced: {error}"
        return " | gripper: open"

    def _apply_sim_pose(self, angles: list[float]) -> None:
        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction
        target = np.array(angles)
        self._sim_arm.set_joint_positions(target)                               # where joints ARE
        self._sim_arm.apply_action(ArticulationAction(joint_positions=target))  # where motors GO

    def _prepare_validation(self, blocks: list[ProgramBlock]) -> None:
        """A simulation run can only validate if it starts at the synced real pose."""
        from . import real_robot
        self._validation = None
        self._pending_validation = None
        self._update_act_button()
        if self._sync_pose is None or self._sim_arm is None:
            return  # plain simulation run, nothing to validate
        try:
            current = [float(a) for a in self._sim_arm.get_joint_positions()]
        except Exception:
            return
        deviation = real_robot.max_deviation_deg(current, self._sync_pose)
        if deviation > real_robot.START_POSE_TOLERANCE_DEG:
            print(f"[Scratch Blocks][real] not validating: simulated arm is {deviation:.1f} deg "
                  "away from the synced pose (press Sync from real first)")
            return
        self._pending_validation = {"fingerprint": real_robot.program_fingerprint(blocks),
                                    "start": list(self._sync_pose)}

    def _on_run_finished(self, succeeded: bool, real: bool) -> None:
        if real:
            asyncio.ensure_future(self._end_real_run(succeeded))
            return
        if succeeded and self._pending_validation is not None:
            self._validation = self._pending_validation
            self._set_status("VALIDATED in simulation - 'Act on real' is unlocked")
        self._pending_validation = None
        self._update_act_button()

    def _act_on_real(self) -> None:
        from . import real_robot
        if self._runner.is_running or self._real_busy:
            self._set_status("Wait until the current run has finished")
            return
        blocks = self._validate_program(report_success=False)
        if blocks is None:
            return
        if self._validation is None:
            self._set_status("Locked: Sync from real, then Run the program in simulation first")
            return
        if real_robot.program_fingerprint(blocks) != self._validation["fingerprint"]:
            self._invalidate("Locked: the program changed since it was simulated - Run it again")
            return
        unsupported = real_robot.unsupported_blocks(blocks)
        if unsupported:
            self._set_status(f"Not on the real robot yet: {', '.join(unsupported)}")
            return
        now = time.monotonic()
        if now > self._act_confirm_until:
            self._act_confirm_until = now + 5.0
            self._set_status("Click 'Act on real' again within 5 s to move the REAL robot")
            return
        self._act_confirm_until = 0.0
        asyncio.ensure_future(self._act_async(blocks))

    async def _act_async(self, blocks: list[ProgramBlock]) -> None:
        from . import real_robot
        from .neurapy_backend import NeuraPyBackend
        from .real_arm_mirror import RealArmMirror
        self._real_busy = True
        started = False
        try:
            self._set_status("Act: checking the real robot is still at the validated start ...")
            if self._real_reader is None:
                self._real_reader = await asyncio.to_thread(real_robot.connect)
            now = [float(a) for a in
                   await asyncio.to_thread(self._real_reader.get_current_joint_angles)]
            deviation = real_robot.max_deviation_deg(now, self._validation["start"])
            if deviation > real_robot.START_POSE_TOLERANCE_DEG:
                self._invalidate(f"Aborted: the real arm moved {deviation:.1f} deg since Sync - "
                                 "Sync and Run again")
                return
            commander = await asyncio.to_thread(real_robot.connect)   # separate command connection
            backend = NeuraPyBackend(commander,
                                     speed_percent=real_robot.REAL_SPEED_PERCENT,
                                     acceleration_percent=real_robot.REAL_ACCEL_PERCENT,
                                     override=real_robot.REAL_OVERRIDE)
            axis_names = tuple(self._sim_arm.dof_names)
            self._mirror = RealArmMirror(self._real_reader, self._sim_arm)
            self._mirror.start()
            # One validation = one real run. Afterwards: Sync + Run again.
            self._validation = None
            self._update_act_button()
            started = self._runner.start(blocks, offline_joint_target=(backend, axis_names))
            if not started:
                self._set_status("Could not start the real run")
            else:
                print("[Scratch Blocks][real] acting on the REAL robot "
                      f"(speed {real_robot.REAL_SPEED_PERCENT:g} %, override {real_robot.REAL_OVERRIDE:g})")
                self._refresh_canvas()
        except Exception as error:
            self._set_status(f"Act failed before moving: {error}")
            carb.log_error(f"[Scratch Blocks][real] act failed: {error}")
        finally:
            if not started:
                self._real_busy = False
                if self._mirror is not None:
                    self._mirror.stop()
                    self._mirror = None

    async def _end_real_run(self, succeeded: bool) -> None:
        await asyncio.sleep(1.0)  # let the mirror show the final pose
        if self._mirror is not None:
            self._mirror.stop()
            self._mirror = None
        self._real_busy = False
        self._sync_pose = None    # the real arm has moved: Sync again before the next run
        self._invalidate()
        last = self._status_label.text if self._status_label is not None else ""
        prefix = "REAL run completed" if succeeded else f"REAL run ended early ({last})"
        self._set_status(prefix + " - Sync from real before the next run")

    def _compile(self) -> None:
        self._validate_program(report_success=True)

    def _validate_program(self, report_success: bool) -> list[ProgramBlock] | None:
        blocks = self._program.execution_blocks()
        if not blocks:
            message = "Connect at least one block to START"
            self._set_status(f"Compile failed: {message}")
            print(f"[Scratch Blocks][Compile Error] {message}")
            return None

        try:
            from .robot_profiles import load_profile, resolve_joint
            profile = None
            repeat_stack = []
            for index, block in enumerate(blocks, start=1):
                if block.type == "repeat":
                    count = float(block.value)
                    if not math.isfinite(count) or not count.is_integer() or not (1 <= count <= 100):
                        raise ValueError("REPEAT count must be a whole number from 1 to 100")
                    repeat_stack.append((index, block.id))
                    continue
                if block.type == "end_repeat":
                    if not repeat_stack:
                        raise ValueError("END REPEAT without REPEAT")
                    opening_index, _ = repeat_stack.pop()
                    if index == opening_index + 1:
                        raise ValueError("REPEAT cannot be empty")
                    continue
                if block.type == "initialize_robot" and repeat_stack:
                    raise ValueError("INIT ROBOT must be outside REPEAT")
                if block.type == "stop_program":
                    continue
                if block.type == "print":
                    if not str(block.value).strip():
                        raise ValueError("PRINT text cannot be empty")
                elif block.type == "set_speed":
                    validate_speed(block.value)
                elif block.type == "wait":
                    seconds = float(block.value)
                    if not math.isfinite(seconds) or seconds < 0:
                        raise ValueError("WAIT requires a finite non-negative duration")
                elif block.type == "initialize_robot":
                    init_value = validate_initialize_parameters(block.value)
                    profile = load_profile(init_value["profile"])
                elif block.type == "initialize_gripper":
                    if profile is None:
                        raise ValueError("INIT GRIPPER requires INIT ROBOT before it")
                    from .model import validate_gripper_init_parameters
                    from .robot_profiles import load_gripper_profile, with_gripper
                    gripper_value = validate_gripper_init_parameters(block.value)
                    profile = with_gripper(profile, load_gripper_profile(gripper_value["profile"]))
                elif block.type in ("robot_properties", "move_all_joints", "move_joint", "move_linear", "move_relative", "gripper", "wave"):
                    if profile is None:
                        raise ValueError(f"{block.type.upper()} requires INIT ROBOT before it")
                    if block.type == "move_all_joints":
                        targets = parse_joint_targets(block.value["targets"])
                        if profile["name"] != "generic":
                            expected = set(profile.get("arm_joint_names", []))
                            if set(targets) != expected:
                                raise ValueError(f"MOVE ALL requires these joints: {sorted(expected)}")
                    elif block.type == "move_joint":
                        if not isinstance(block.value, dict):
                            raise ValueError("MOVE JOINT requires parameters")
                        joint_name = str(block.value.get("joint_name", "")).strip()
                        if not joint_name:
                            raise ValueError("MOVE JOINT requires a joint name")
                        if profile["name"] != "generic":
                            resolve_joint(profile, joint_name)
                        if not math.isfinite(float(block.value["target_degrees"])):
                            raise ValueError("MOVE JOINT target must be finite")
                    elif block.type in ("move_linear", "move_relative"):
                        value = validate_linear_parameters(block.value)
                        descriptor, urdf = value["robot_description_path"], value["urdf_path"]
                        if bool(descriptor) != bool(urdf):
                            raise ValueError("Provide both YAML and URDF, or use INIT profile")
                        if descriptor and not value["end_effector_frame"]:
                            raise ValueError("Explicit kinematics files require a frame")
                        if not descriptor and not profile.get("kinematics"):
                            raise ValueError("INIT profile has no linear kinematics")
                    elif block.type == "wave":
                        value = validate_wave_parameters(block.value)
                        if not value["joint_name"] or value["joint_name"] == "Select joint":
                            raise ValueError("WAVE: choose a joint in the block")
                        if profile["name"] != "generic":
                            resolve_joint(profile, value["joint_name"])
                        from .real_robot import max_joint_speed_deg_s
                        amplitude = (value["high"] - value["low"]) / 2.0
                        limit = max_joint_speed_deg_s(value["joint_name"])
                        # Same shape factor as the simulation (robot_control.WAVE_SHARPNESS).
                        from .robot_control import WAVE_SHARPNESS as s
                        factor = s / math.asin(s) if s > 1e-6 else 1.0
                        peak = 2.0 * math.pi * value["frequency"] * amplitude * factor
                        if peak > limit:
                            f_max = limit / (2.0 * math.pi * amplitude * factor)
                            raise ValueError(
                                f"WAVE {value['joint_name']}: {peak:.0f} deg/s is above the real arm's "
                                f"{limit:g} deg/s (datasheet) - use at most {f_max:.2f} Hz or a smaller swing")
                    elif block.type == "gripper":
                        if not profile.get("gripper"):
                            raise ValueError("No gripper: add INIT GRIPPER after INIT ROBOT")
                        if str(block.value).strip().lower() not in ("open", "close"):
                            raise ValueError("GRIPPER action must be open or close")
                else:
                    raise ValueError(f"Unsupported block type: {block.type}")
            if repeat_stack:
                raise ValueError("REPEAT is missing END REPEAT")
        except (TypeError, ValueError, KeyError, OverflowError, OSError) as error:
            self._set_status(f"Compile failed: {error}")
            print(f"[Scratch Blocks][Compile Error] {error}")
            return None

        if report_success:
            self._set_status(f"Compile successful: {len(blocks)} blocks")
            print(f"[Scratch Blocks] Compile successful: {len(blocks)} block(s)")
            for index, block in enumerate(blocks, start=1):
                print(f"[Scratch Blocks]   {index}: {block.type.upper()} {block.value}")
        return blocks

    def _clear(self) -> None:
        if self._runner.is_running:
            return
        self._program.clear()
        self._selected.clear()
        self._invalidate_scene_cache()
        self._speed_models.clear()
        self._joint_degree_models.clear()
        self._all_joint_models.clear()
        self._linear_models.clear()
        self._wave_models.clear()
        self._refresh_canvas()
        self._set_status("Cleared")

    @staticmethod
    def _safe_project_name(raw: str) -> str:
        import re
        name = re.sub(r"[^A-Za-z0-9_\- ]+", "_", str(raw)).strip(" _")
        return name or "my_program"

    def _init_robot_path(self) -> str | None:
        init = next((b for b in self._program.blocks.values() if b.type == "initialize_robot"), None)
        if init is None:
            return None
        try:
            return validate_initialize_parameters(init.value)["robot_path"]
        except ValueError:
            return None

    def _save(self) -> None:
        """Save under the name in the Project field (Documents/ScratchRobotProjects)."""
        name = self._safe_project_name(self._project_name_model.get_value_as_string())
        self._project_name_model.set_value(name)
        path = os.path.join(self._projects_dir, f"{name}.json")
        try:
            os.makedirs(self._projects_dir, exist_ok=True)
            existed = os.path.exists(path)
            data = self._program.to_dict()
            # Extra information for humans and for opening on another robot.
            data["project"] = {"name": name,
                               "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "robot_path": self._init_robot_path()}
            with open(path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2)
            self._set_status(f"Saved '{name}'" + (" (replaced the older version)" if existed else "")
                             + f" in {self._projects_dir}")
        except Exception as error:
            self._set_status(f"Save failed: {error}")

    def _project_files(self) -> list[str]:
        files = []
        if os.path.isdir(self._projects_dir):
            files = [os.path.join(self._projects_dir, f) for f in os.listdir(self._projects_dir)
                     if f.lower().endswith(".json")]
        files.sort(key=os.path.getmtime, reverse=True)       # newest first
        if os.path.exists(self._save_path):                   # the old single save slot
            files.append(self._save_path)
        return files

    def _open_menu(self) -> None:
        if self._runner.is_running:
            return
        files = self._project_files()
        self._open_menu_widget = ui.Menu("Open project")
        with self._open_menu_widget:
            if not files:
                ui.MenuItem("No saved projects yet", enabled=False)
            for path in files[:30]:
                stamp = time.strftime("%d.%m. %H:%M", time.localtime(os.path.getmtime(path)))
                label = ("(old save slot)" if path == self._save_path
                         else os.path.splitext(os.path.basename(path))[0])
                ui.MenuItem(f"{label}   [{stamp}]",
                            triggered_fn=lambda p=path: self._load_file(p))
            ui.Separator()
            ui.MenuItem("Open projects folder", triggered_fn=self._show_projects_folder)
            ui.MenuItem(f"Change projects folder...  (now: {self._projects_dir})",
                        triggered_fn=self._change_folder_dialog)
        self._open_menu_widget.show()

    @staticmethod
    def _load_projects_dir() -> str:
        try:
            import carb.settings
            saved = carb.settings.get_settings().get(PROJECTS_DIR_SETTING)
            if isinstance(saved, str) and saved.strip():
                return saved
        except Exception:
            pass
        return PROJECTS_DIR

    def _set_projects_dir(self, folder: str) -> bool:
        """Use a new projects folder; creates it if needed and remembers it."""
        folder = os.path.abspath(os.path.expanduser(str(folder).strip().strip('"')))
        try:
            os.makedirs(folder, exist_ok=True)
            probe = os.path.join(folder, ".write_test")
            with open(probe, "w") as handle:   # make sure we can actually save there
                handle.write("ok")
            os.remove(probe)
        except Exception as error:
            self._set_status(f"Can't use that folder: {error}")
            return False
        self._projects_dir = folder
        try:
            import carb.settings
            carb.settings.get_settings().set(PROJECTS_DIR_SETTING, folder)
        except Exception:
            pass
        self._set_status(f"Projects folder: {folder}")
        return True

    def _change_folder_dialog(self) -> None:
        """Small window: paste a folder path (e.g. copied from Explorer's address bar)."""
        model = ui.SimpleStringModel(self._projects_dir)
        self._folder_window = ui.Window("Projects folder", width=560, height=120)
        with self._folder_window.frame:
            with ui.VStack(spacing=6, style={"margin": 4}):
                ui.Label("Folder for saved projects (paste a path from Explorer's address bar):")
                ui.StringField(model, height=24)
                with ui.HStack(height=26, spacing=6):
                    def use():
                        if self._set_projects_dir(model.get_value_as_string()):
                            self._folder_window.visible = False
                    def default():
                        model.set_value(PROJECTS_DIR)
                    ui.Button("Use this folder", clicked_fn=use)
                    ui.Button("Reset to default", clicked_fn=default)
                    ui.Button("Cancel", clicked_fn=lambda: setattr(self._folder_window, "visible", False))

    def _show_projects_folder(self) -> None:
        os.makedirs(self._projects_dir, exist_ok=True)
        try:
            os.startfile(self._projects_dir)   # Windows Explorer
        except Exception:
            self._set_status(f"Projects folder: {self._projects_dir}")

    def _load(self) -> None:
        """Old single-slot load, kept for compatibility."""
        self._load_file(self._save_path)

    def _load_file(self, path: str) -> None:
        if self._runner.is_running:
            return
        try:
            with open(path, "r", encoding="utf-8") as file:
                self._program.load_dict(json.load(file))
            self._selected.clear()
            self._invalidate_scene_cache()
            self._speed_models.clear()
            self._all_joint_models.clear()
            self._joint_degree_models.clear()
            self._linear_models.clear()
            self._wave_models.clear()
            self._invalidate()                 # a different program is never validated
            self._refresh_canvas()
            if path != self._save_path:
                self._project_name_model.set_value(os.path.splitext(os.path.basename(path))[0])
            robot_path = self._init_robot_path()
            if robot_path and not self._prim_exists(robot_path):
                self._set_status(f"Opened - but {robot_path} is not in this scene: pick your robot "
                                 "in INIT ROBOT, joints are matched by number")
            else:
                self._set_status(f"Opened {os.path.basename(path)}")
        except Exception as error:
            self._set_status(f"Open failed: {error}")

    @staticmethod
    def _prim_exists(path: str) -> bool:
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            return bool(stage and stage.GetPrimAtPath(path).IsValid())
        except Exception:
            return True   # can't check: don't raise a false alarm

    def _set_status(self, message: str) -> None:
        if self._status_label is not None:
            self._status_label.text = message

    def _set_active_block(self, block_id: str | None) -> None:
        """Move the yellow running-block frame without rebuilding the canvas."""
        previous = self._active_block_id
        self._active_block_id = block_id
        if previous == block_id:
            return
        old_rect = self._highlight_rects.get(previous) if previous else None
        if old_rect is not None:
            old_rect.visible = False
        new_rect = self._highlight_rects.get(block_id) if block_id else None
        if new_rect is not None:
            new_rect.visible = True
        if block_id is None:
            # Program finished: unlock drag handles once the runner has exited.
            asyncio.ensure_future(self._refresh_next_frame())

    def _refresh_canvas(self) -> None:
        if self._canvas_frame is not None:
            self._canvas_frame.rebuild()
