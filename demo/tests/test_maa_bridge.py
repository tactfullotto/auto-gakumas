"""
Milestone 3 Test Suite: MaaFramework Bridge, Controllers, Pipeline, and Automation Runner.
Covers:
- Tier 1: Functional baseline for BaseController, MockController, AdbController, Win32Controller, Pipeline.json
- Tier 2: Boundary values, coordinate clamping, timeout handling, error recovery, scenario injection
- Tier 3: Pairwise integration of CustomRecognition, CustomAction, and GakumasRunner end-to-end loop
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from demo.core.db import GakumasDB
from demo.schemas import BoundingBox, DecisionResult, LegalAction, ScreenObservation
from demo.vision.mock_generator import MockFrameGenerator
from demo.vision.roi import BASE_HEIGHT, BASE_WIDTH
from demo.agent.client import AgentConfig, AgentDecisionMaker
from demo.maa_bridge.base_controller import BaseController, ControllerExecutionError
from demo.maa_bridge.mock_controller import MockController
from demo.maa_bridge.adb_controller import AdbConfig, AdbController
from demo.maa_bridge.win32_controller import Win32Controller
from demo.maa_bridge.custom_recognition import (
    GakumasBattleRecognition,
    GakumasProduceRecognition,
    GakumasSceneRoutingRecognition,
    CustomRecognitionResult,
    Rect,
)
from demo.maa_bridge.custom_action import GakumasDecisionAction
from demo.maa_bridge.simulator import MaaPipelineSimulator
from demo.maa_bridge.runner import GakumasRunner, StepResult, RunnerState


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(scope="module")
def shared_db():
    db = GakumasDB()
    yield db
    db.close()


@pytest.fixture
def mock_agent():
    return AgentDecisionMaker(AgentConfig(api_key=None, fallback_on_error=True))


@pytest.fixture
def sample_frames(shared_db):
    gen = MockFrameGenerator(db=shared_db, plan="sense")
    battle_frame, battle_obs = gen.generate_battle_frame(plan="sense")
    produce_frame, produce_obs = gen.generate_produce_schedule_frame(plan="sense")
    popup_frame, popup_obs = gen.generate_popup_frame()
    return {
        "battle": (battle_frame, battle_obs),
        "produce": (produce_frame, produce_obs),
        "popup": (popup_frame, popup_obs),
    }


# ==============================================================================
# 1. BaseController & MockController Tests (Tier 1 & 2)
# ==============================================================================

def test_base_controller_abstract_contract():
    """Verify BaseController cannot be instantiated directly without implementations."""
    with pytest.raises(TypeError):
        BaseController()  # type: ignore


def test_mock_controller_connection_lifecycle(shared_db):
    """Verify connect, disconnect, and is_connected state transitions."""
    controller = MockController(db=shared_db, plan="sense")
    assert not controller.is_connected
    assert controller.connect() is True
    assert controller.is_connected is True
    controller.disconnect()
    assert controller.is_connected is False


def test_mock_controller_screencap_format(shared_db):
    """Verify screencap returns valid 1080x1920 uint8 frame."""
    controller = MockController(db=shared_db, plan="sense")
    controller.connect()
    frame = controller.screencap()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (BASE_HEIGHT, BASE_WIDTH, 3)
    assert frame.dtype == np.uint8


def test_mock_controller_screencap_disconnected_raises(shared_db):
    """Verify screencap raises ControllerExecutionError when disconnected."""
    controller = MockController(db=shared_db, plan="sense")
    with pytest.raises(ControllerExecutionError):
        controller.screencap()


def test_mock_controller_click_recording(shared_db):
    """Verify clicks are recorded with exact coordinates and timestamps."""
    controller = MockController(db=shared_db, plan="sense")
    controller.connect()
    controller.click(540, 960)
    controller.click(100, 200)
    history = controller.get_click_history()
    assert len(history) == 2
    assert history[0][0] == "click"
    assert history[0][1:3] == (540, 960)
    assert history[1][1:3] == (100, 200)


def test_mock_controller_swipe_recording(shared_db):
    """Verify swipe actions are recorded accurately."""
    controller = MockController(db=shared_db, plan="sense")
    controller.connect()
    controller.swipe(100, 500, 100, 200, duration_ms=150)
    history = controller.get_click_history()
    assert len(history) == 1
    assert history[0][0] == "swipe"
    assert history[0][1:5] == (100, 500, 100, 200)


def test_mock_controller_scenario_injection(shared_db, sample_frames):
    """Verify injecting scenarios sets custom screencap output."""
    controller = MockController(db=shared_db, plan="sense")
    controller.connect()
    battle_frame, _ = sample_frames["battle"]
    produce_frame, _ = sample_frames["produce"]

    controller.inject_frame(battle_frame)
    cap1 = controller.screencap()
    assert np.array_equal(cap1, battle_frame)

    controller.inject_frame(produce_frame)
    # Trigger click to advance frame queue
    controller.click(540, 960)
    cap2 = controller.screencap()
    assert np.array_equal(cap2, produce_frame)


def test_mock_controller_boundary_click_clamping(shared_db):
    """Tier 2 BVA: coordinates at boundary (0,0), (1079,1919), and out-of-screen accepted without crashing."""
    controller = MockController(db=shared_db, plan="sense")
    controller.connect()
    assert controller.click(0, 0) is True
    assert controller.click(1079, 1919) is True
    assert controller.click(-10, -20) is True
    assert controller.click(2000, 3000) is True


def test_mock_controller_hit_testing(shared_db):
    """Verify last_click_hit_bbox accurately matches coordinates within bounding box."""
    controller = MockController(db=shared_db, plan="sense")
    controller.connect()
    box = BoundingBox(xmin=100, ymin=200, xmax=300, ymax=400)

    controller.click(150, 250)
    assert controller.last_click_hit_bbox(box) is True

    controller.click(50, 50)
    assert controller.last_click_hit_bbox(box) is False


def test_mock_controller_context_manager(shared_db):
    """Verify MockController functions cleanly as a context manager."""
    with MockController(db=shared_db, plan="sense") as ctrl:
        assert ctrl.is_connected is True
        frame = ctrl.screencap()
        assert frame is not None
    assert ctrl.is_connected is False


# ==============================================================================
# 2. AdbController Tests (Tier 1 & 2)
# ==============================================================================

def test_adb_controller_defaults():
    """Verify default connection address is 192.168.0.115:5555."""
    controller = AdbController()
    assert controller.host == "192.168.0.115"
    assert controller.port == 5555


@patch("subprocess.run")
@patch("socket.socket")
def test_adb_controller_connect_success_mocked(mock_socket, mock_run):
    """Verify successful adb connect parses output and sets connected."""
    mock_run.return_value = MagicMock(returncode=0, stdout="connected to 192.168.0.115:5555\n", stderr="")
    controller = AdbController(host="192.168.0.115", port=5555)
    assert controller.connect() is True
    assert controller.is_connected is True


@patch("subprocess.run")
@patch("socket.socket")
def test_adb_controller_connect_failure_graceful(mock_socket, mock_run):
    """Verify connection timeout/failure returns False without crashing."""
    mock_run.side_effect = TimeoutError("Connection timed out")
    controller = AdbController(host="192.168.0.115", port=5555)
    assert controller.connect() is False
    assert controller.is_connected is False


@patch("subprocess.run")
def test_adb_controller_click_command_formatting(mock_run):
    """Verify click translates to 'adb shell input tap x y'."""
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    controller = AdbController()
    controller.is_connected = True
    assert controller.click(500, 800) is True

    called_cmd = mock_run.call_args[0][0]
    assert "input" in called_cmd
    assert "tap" in called_cmd
    assert "500" in called_cmd and "800" in called_cmd


@patch("subprocess.run")
def test_adb_controller_swipe_command_formatting(mock_run):
    """Verify swipe translates to 'adb shell input swipe x1 y1 x2 y2 duration'."""
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    controller = AdbController()
    controller.is_connected = True
    assert controller.swipe(100, 200, 300, 400, duration_ms=250) is True

    called_cmd = mock_run.call_args[0][0]
    assert "input" in called_cmd
    assert "swipe" in called_cmd
    assert "100" in called_cmd and "200" in called_cmd and "250" in called_cmd


@patch("subprocess.run")
def test_adb_controller_screencap_reconnects_on_transient_error(mock_run):
    """Verify screencap issues adb connect and retries once after non-zero exec-out exit."""
    import io
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (120, 200), (12, 34, 56)).save(buf, format="PNG")
    valid_png = buf.getvalue()

    mock_run.side_effect = [
        MagicMock(returncode=1, stdout=b"", stderr=b"device not found"),
        MagicMock(returncode=0, stdout="connected to 192.168.0.115:5555\n", stderr=""),
        MagicMock(returncode=0, stdout=valid_png, stderr=b""),
    ]
    controller = AdbController(AdbConfig(prefer_maa=False, fallback_to_mock=False))
    controller.is_connected = True

    frame = controller.screencap()
    assert frame.shape == (200, 120, 3)
    assert mock_run.call_count == 3


@patch("subprocess.run")
def test_adb_controller_screencap_fallback_to_mock_on_cli_error(mock_run):
    """Verify fallback_to_mock delegates cleanly even when exec-out exits non-zero without raising."""
    mock_run.side_effect = [
        MagicMock(returncode=1, stdout=b"", stderr=b"offline"),
        MagicMock(returncode=1, stdout="failed to connect", stderr=""),
    ]
    controller = AdbController(AdbConfig(prefer_maa=False, fallback_to_mock=True))
    controller.is_connected = True

    frame = controller.screencap()
    assert frame is not None and frame.shape == (BASE_HEIGHT, BASE_WIDTH, 3)


# ==============================================================================
# 3. Win32Controller Tests (Tier 1 & 2)
# ==============================================================================

def test_win32_controller_defaults():
    """Verify default window title and settings."""
    controller = Win32Controller()
    assert "学园偶像大师" in controller.window_title or "Gakumas" in controller.window_title


@patch("ctypes.windll.user32.FindWindowW")
def test_win32_controller_connect_found(mock_find):
    """Verify window found sets hwnd and is_connected."""
    mock_find.return_value = 0x12345
    controller = Win32Controller()
    assert controller.connect() is True
    assert controller.is_connected is True


@patch("ctypes.windll.user32.FindWindowW")
def test_win32_controller_connect_not_found_fallback(mock_find):
    """Verify window not found returns False gracefully."""
    mock_find.return_value = 0
    controller = Win32Controller()
    assert controller.connect() is False
    assert controller.is_connected is False


def test_win32_controller_headless_fallback():
    """Verify headless fallback returns True and produces screencap when window missing."""
    controller = Win32Controller(headless_fallback=True)
    with patch("ctypes.windll.user32.FindWindowW", return_value=0):
        assert controller.connect() is True
        assert controller.is_connected is True
        frame = controller.screencap()
        assert frame is not None
        assert frame.shape == (BASE_HEIGHT, BASE_WIDTH, 3)


# ==============================================================================
# 4. Pipeline.json Syntax & Graph Consistency Tests (Tier 1 & 2)
# ==============================================================================

def test_pipeline_json_exists_and_valid():
    """Verify pipeline.json exists and parses as valid JSON."""
    pipeline_path = Path("demo/maa_bridge/pipeline.json")
    assert pipeline_path.exists(), "demo/maa_bridge/pipeline.json must exist."
    with open(pipeline_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_pipeline_json_entry_and_nodes():
    """Verify pipeline.json contains required entry and routing nodes."""
    pipeline_path = Path("demo/maa_bridge/pipeline.json")
    with open(pipeline_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "SceneRouting" in data
    expected_nodes = [
        "SceneRouting",
        "BattleTurn",
        "ProduceSchedule",
        "CardReward",
        "Shop",
        "PopupConfirmation",
        "SkipCutscene",
    ]
    for node in expected_nodes:
        assert node in data, f"Required node '{node}' missing from pipeline.json"

    # Check graph reachability: every target in 'next' must exist in data
    all_nodes = set(data.keys())
    for node_name, node_body in data.items():
        if "next" in node_body and isinstance(node_body["next"], list):
            for target in node_body["next"]:
                assert target in all_nodes, f"Node '{node_name}' references missing target '{target}'"
        if "on_error" in node_body and isinstance(node_body["on_error"], list):
            for target in node_body["on_error"]:
                assert target in all_nodes, f"Node '{node_name}' on_error references missing target '{target}'"


# ==============================================================================
# 5. CustomRecognition & CustomAction Bridge Tests (Tier 2 & 3)
# ==============================================================================

def test_custom_recognition_battle_frame(shared_db, sample_frames):
    """Verify CustomRecognition correctly extracts ScreenObservation from frame."""
    frame, gt_obs = sample_frames["battle"]
    reco = GakumasBattleRecognition(db=shared_db, plan="sense")
    obs = reco.recognize(frame)
    assert obs is not None
    assert obs.scene_type == "battle_turn"
    assert len(obs.cards) == len(gt_obs.cards)


def test_custom_recognition_scene_routing(shared_db, sample_frames):
    """Verify GakumasSceneRoutingRecognition classifies all synthetic frames accurately."""
    reco = GakumasSceneRoutingRecognition(db=shared_db, plan="sense")

    battle_obs = reco.recognize(sample_frames["battle"][0])
    assert battle_obs is not None
    assert battle_obs.scene_type == "battle_turn"

    produce_obs = reco.recognize(sample_frames["produce"][0])
    assert produce_obs is not None
    assert produce_obs.scene_type == "produce_schedule"

    popup_obs = reco.recognize(sample_frames["popup"][0])
    assert popup_obs is not None
    assert popup_obs.scene_type == "popup"


def test_custom_action_maps_choice_id_to_bbox_center(shared_db):
    """Verify CustomAction executes click on the center of chosen action's bbox."""
    controller = MockController(db=shared_db, plan="sense")
    controller.connect()

    action_box = BoundingBox(xmin=100, ymin=200, xmax=300, ymax=400)
    legal_action = LegalAction(
        choice_id=0,
        action_type="play_card",
        description="Test Action",
        bbox=action_box,
    )
    decision = DecisionResult(choice_id=0, reason="Test Reason", source="fallback")

    custom_action = GakumasDecisionAction(controller=controller)
    success = custom_action.execute_decision(decision, [legal_action])
    assert success is True

    history = controller.get_click_history()
    assert len(history) == 1
    assert history[0][1:3] == action_box.center() == (200, 300)


def test_simulator_pipeline_execution(shared_db, sample_frames):
    """Verify MaaPipelineSimulator executes pipeline.json nodes with mock controller."""
    controller = MockController(db=shared_db, plan="sense")
    controller.inject_frame(sample_frames["battle"][0])

    sim = MaaPipelineSimulator("demo/maa_bridge/pipeline.json")
    sim.register_custom_recognition("GakumasSceneRoutingRecognition", GakumasSceneRoutingRecognition(db=shared_db))
    sim.register_custom_recognition("GakumasBattleRecognition", GakumasBattleRecognition(db=shared_db))
    sim.register_custom_action("GakumasDecisionAction", GakumasDecisionAction(controller=controller))
    sim.bind_controller(controller)

    step_res = sim.step()
    assert step_res.step == 1
    assert step_res.recognition_hit is True


# ==============================================================================
# 6. Automation Runner End-to-End Tests (Tier 2 & 3)
# ==============================================================================

def test_runner_initialization_modes(shared_db):
    """Verify runner instantiates correct controllers for all modes."""
    runner_mock = GakumasRunner(mode="mock", db=shared_db)
    assert isinstance(runner_mock.controller, MockController)

    runner_adb = GakumasRunner(mode="adb", db=shared_db)
    assert isinstance(runner_adb.controller, AdbController)

    runner_win32 = GakumasRunner(mode="win32", db=shared_db)
    assert isinstance(runner_win32.controller, Win32Controller)


def test_runner_run_once_battle_step(shared_db, mock_agent, sample_frames):
    """Verify single step cycle: frame -> perception -> prompt -> decision -> click."""
    controller = MockController(db=shared_db, plan="sense")
    battle_frame, _ = sample_frames["battle"]
    controller.inject_frame(battle_frame)

    runner = GakumasRunner(
        mode="mock",
        controller=controller,
        agent=mock_agent,
        db=shared_db,
    )
    assert runner.start() is True

    step_res = runner.run_once()
    assert step_res.success is True
    assert step_res.scene_type == "battle_turn"
    assert step_res.decision is not None
    assert step_res.action_taken is not None
    assert step_res.click_coords is not None
    assert len(controller.get_click_history()) == 1


def test_runner_callbacks_invoked(shared_db, mock_agent, sample_frames):
    """Verify all Rich TUI callback hooks are invoked in expected order."""
    events = []

    def on_obs(obs):
        events.append(("obs", obs.scene_type))

    def on_prompt(prompt, actions):
        events.append(("prompt", len(actions)))

    def on_dec(dec, act):
        events.append(("dec", dec.choice_id))

    def on_act(act, coords):
        events.append(("act", coords))

    def on_step(res):
        events.append(("step", res.success))

    controller = MockController(db=shared_db, plan="sense")
    battle_frame, _ = sample_frames["battle"]
    controller.inject_frame(battle_frame)

    runner = GakumasRunner(
        mode="mock",
        controller=controller,
        agent=mock_agent,
        db=shared_db,
        on_observation=on_obs,
        on_prompt_built=on_prompt,
        on_decision=on_dec,
        on_action_executed=on_act,
        on_step_completed=on_step,
    )
    runner.start()
    step_res = runner.run_once()

    event_names = [e[0] for e in events]
    assert event_names == ["obs", "prompt", "dec", "act", "step"]


def test_runner_run_loop_multi_step_scene_transitions(shared_db, mock_agent, sample_frames):
    """Tier 3: Multi-step runner loop navigating produce_schedule -> battle_turn -> popup."""
    controller = MockController(db=shared_db, plan="sense")
    controller.inject_frame(sample_frames["produce"][0])
    controller.inject_frame(sample_frames["battle"][0])
    controller.inject_frame(sample_frames["popup"][0])

    runner = GakumasRunner(
        mode="mock",
        controller=controller,
        agent=mock_agent,
        db=shared_db,
    )
    runner.start()

    results = runner.run_loop(max_steps=3, delay_between_steps=0.0)
    assert len(results) == 3
    assert [r.scene_type for r in results] == ["produce_schedule", "battle_turn", "popup"]


def test_runner_run_loop_stop_on_scene(shared_db, mock_agent, sample_frames):
    """Verify run_loop terminates when specified stop_on_scene is encountered."""
    controller = MockController(db=shared_db, plan="sense")
    controller.inject_frame(sample_frames["produce"][0])
    controller.inject_frame(sample_frames["popup"][0])
    controller.inject_frame(sample_frames["battle"][0])

    runner = GakumasRunner(mode="mock", controller=controller, agent=mock_agent, db=shared_db)
    runner.start()

    results = runner.run_loop(max_steps=10, stop_on_scene="popup", delay_between_steps=0.0)
    assert len(results) == 2
    assert results[-1].scene_type == "popup"


def test_runner_disconnect_recovery(shared_db, mock_agent, sample_frames):
    """Verify runner recovers when controller is disconnected before run_once."""
    controller = MockController(db=shared_db, plan="sense")
    controller.inject_frame(sample_frames["battle"][0])

    runner = GakumasRunner(mode="mock", controller=controller, agent=mock_agent, db=shared_db)
    runner.start()

    # Artificially disconnect
    controller.disconnect()
    assert not controller.is_connected

    # run_once should trigger auto-reconnect
    res = runner.run_once()
    assert res.success is True
    assert controller.is_connected is True


def test_runner_exception_safety(shared_db, sample_frames):
    """Verify runner wraps unexpected agent exceptions into StepResult with success=False."""
    controller = MockController(db=shared_db, plan="sense")
    controller.inject_frame(sample_frames["battle"][0])

    faulty_agent = MagicMock()
    faulty_agent.decide.side_effect = RuntimeError("Simulated agent engine crash")

    runner = GakumasRunner(mode="mock", controller=controller, agent=faulty_agent, db=shared_db)
    runner.start()

    res = runner.run_once()
    assert res.success is False
    assert "Simulated agent engine crash" in (res.error or "")


# ==============================================================================
# 7. Remediation Tests (Simulator Candidate Fallback, Node Name, Controller Guards)
# ==============================================================================

def test_simulator_candidate_fallback_evaluation():
    """Verify that when candidate 1 fails recognition, candidate 2 is evaluated and executed."""
    cand1_called = False
    cand2_called = False

    class Cand1:
        def analyze(self, ctx, arg):
            nonlocal cand1_called
            cand1_called = True
            return None

    class Cand2:
        def analyze(self, ctx, arg):
            nonlocal cand2_called
            cand2_called = True
            return CustomRecognitionResult(Rect(0, 0, 10, 10), "{}")

    pipeline = {
        "SceneRouting": {
            "action": "DoNothing",
            "next": ["C1", "C2"],
        },
        "C1": {
            "recognition": "Custom",
            "custom_recognition": "Cand1",
            "action": "DoNothing",
            "on_error": ["SceneRouting"],
        },
        "C2": {
            "recognition": "Custom",
            "custom_recognition": "Cand2",
            "action": "DoNothing",
            "next": ["SceneRouting"],
        },
    }

    sim = MaaPipelineSimulator(pipeline)
    sim.register_custom_recognition("Cand1", Cand1())
    sim.register_custom_recognition("Cand2", Cand2())

    # Step 1: executes SceneRouting, next candidate list becomes ["C1", "C2"]
    res1 = sim.step()
    assert res1.step == 1
    assert res1.node_name == "SceneRouting"
    assert res1.recognition_hit is True
    assert sim.current_node == "C1"

    # Step 2: evaluates C1 (returns None), falls back to C2 (returns result)
    res2 = sim.step()
    assert res2.step == 2
    assert cand1_called is True
    assert cand2_called is True
    assert res2.node_name == "C2"
    assert res2.recognition_hit is True
    assert res2.next_node == "SceneRouting"


def test_simulator_step_result_node_name():
    """Verify PipelineStepResult.node_name reflects the executing node and not next_node."""
    pipeline = {
        "NodeA": {
            "action": "DoNothing",
            "next": ["NodeB"],
        },
        "NodeB": {
            "action": "DoNothing",
            "next": ["NodeA"],
        },
    }
    sim = MaaPipelineSimulator(pipeline)
    sim.current_node = "NodeA"

    res1 = sim.step()
    assert res1.node_name == "NodeA"
    assert res1.next_node == "NodeB"
    assert sim.current_node == "NodeB"

    res2 = sim.step()
    assert res2.node_name == "NodeB"
    assert res2.next_node == "NodeA"
    assert sim.current_node == "NodeA"


def test_win32_controller_float_coordinates_and_malformed_regex():
    """Verify Win32Controller safely handles float coordinates and malformed regex."""
    controller = Win32Controller(headless_fallback=True)
    assert controller.connect() is True
    # Float coordinates should cast to int without bitshift TypeError
    assert controller.click(500.7, 800.2) is True

    # Malformed regex in _find_window_by_pattern should return None without raising re.error
    assert controller._find_window_by_pattern("[invalid(regex") is None


@patch("subprocess.run")
def test_adb_controller_float_coordinates_and_timeout(mock_run):
    """Verify AdbController handles float coordinates and subprocess.TimeoutExpired."""
    import subprocess
    controller = AdbController()
    controller.is_connected = True

    # Successful float click
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    assert controller.click(123.4, 567.8) is True
    cmd = mock_run.call_args[0][0]
    assert cmd[-2:] == ["123", "567"]

    # Click timeout
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["adb"], timeout=5)
    assert controller.click(100, 200) is False

    # Swipe float coordinates and timeout
    mock_run.side_effect = None
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    assert controller.swipe(10.1, 20.2, 30.3, 40.4, duration_ms=250.0) is True
    swipe_cmd = mock_run.call_args[0][0]
    assert swipe_cmd[-5:] == ["10", "20", "30", "40", "250"]

    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["adb"], timeout=5)
    assert controller.swipe(10, 20, 30, 40) is False


def test_mock_controller_stamina_clamp_at_zero(shared_db):
    """Verify that card click with 0 stamina clamps to 0 rather than resetting to 1."""
    controller = MockController(db=shared_db, plan="sense", auto_progress=True)
    controller.connect()

    # Set scenario with stamina=0 and cards
    controller.set_scenario(scene_type="battle_turn", stamina=0, turn=1)
    # Click card 0
    controller.click(150, 1500)
    assert controller.current_obs is not None
    assert controller.current_obs.stamina == 0


def test_custom_recognition_saturated_white_frame_underflow_guard():
    """Verify classify_scene does not falsely detect popup on saturated white frame."""
    from demo.maa_bridge.custom_recognition import classify_scene
    white_frame = np.full((1920, 1080, 3), 255, dtype=np.uint8)
    scene = classify_scene(white_frame)
    assert scene != "popup"
