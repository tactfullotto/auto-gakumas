"""
Integration & API tests for the MFAAvalonia-style Gakumas Web UI server (demo/web/).
"""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from demo.app import build_arg_parser
from demo.web.server import create_web_server


@pytest.fixture(scope="module")
def web_server():
    httpd, manager = create_web_server(host="127.0.0.1", port=0, auto_preview=True)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        yield base_url, manager
    finally:
        manager.stop_all()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3.0)


def _get_bytes(url: str):
    with urllib.request.urlopen(url, timeout=5.0) as resp:
        return resp.status, dict(resp.headers), resp.read()


def _post_json(url: str, payload: dict):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10.0) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_app_parser_supports_ui_mode():
    parser = build_arg_parser()
    args = parser.parse_args(["--mode", "ui", "--ui-port", "9001"])
    assert args.mode == "ui"
    assert args.ui_port == 9001


def test_static_pages_served(web_server):
    base_url, _ = web_server

    status, headers, html = _get_bytes(f"{base_url}/")
    assert status == 200
    assert "text/html" in headers.get("Content-Type", "")
    assert b"GakumasMAA" in html

    status_css, _, css = _get_bytes(f"{base_url}/static/styles.css")
    assert status_css == 200
    assert b"color-scheme" in css

    status_js, _, js = _get_bytes(f"{base_url}/static/app.js")
    assert status_js == 200
    assert b"/api/state" in js


def test_initial_state_and_png_frame(web_server):
    base_url, _ = web_server

    status, _, raw = _get_bytes(f"{base_url}/api/state")
    assert status == 200
    data = json.loads(raw.decode("utf-8"))
    assert data["active_id"] == "default"
    assert len(data["instances"]) >= 3
    active = data["active"]
    assert active["observation"] is not None
    assert active["est_tokens"] > 100
    assert "一、剧本流程" in active["prompt_text"]
    assert "scenario_roadmap" in active
    assert len(active["scenario_roadmap"]["weeks"]) == 13
    assert "llm_api_key" not in active["config"]

    sc_status, _, sc_raw = _get_bytes(f"{base_url}/api/scenarios")
    assert sc_status == 200
    sc_catalog = json.loads(sc_raw.decode("utf-8"))
    assert "hajime_regular" in sc_catalog
    assert sc_catalog["hajime_regular"]["total_weeks"] == 13

    f_status, f_headers, f_bytes = _get_bytes(f"{base_url}/api/frame?instance_id=default")
    assert f_status == 200
    assert f_headers.get("Content-Type") == "image/png"
    assert f_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_library_endpoints_and_asset_guard(web_server):
    base_url, _ = web_server

    status, _, raw_cards = _get_bytes(f"{base_url}/api/library/cards?plan=sense&limit=5")
    assert status == 200
    cards_data = json.loads(raw_cards.decode("utf-8"))
    assert cards_data["total"] > 0
    assert len(cards_data["items"]) <= 5

    first_icon = next((c["icon_url"] for c in cards_data["items"] if c.get("icon_url")), None)
    assert first_icon is not None
    icon_status, icon_headers, icon_bytes = _get_bytes(f"{base_url}{first_icon}")
    assert icon_status == 200
    assert "image/" in icon_headers.get("Content-Type", "")
    assert len(icon_bytes) > 100

    d_status, _, raw_drinks = _get_bytes(f"{base_url}/api/library/drinks?plan=logic&limit=5")
    assert d_status == 200
    drinks_data = json.loads(raw_drinks.decode("utf-8"))
    assert drinks_data["total"] > 0

    for blocked in (
        "/assets/manifest.json",
        "/assets/../manifest.json",
        "/assets/cards/../../demo/schemas.py",
        "/static/../server.py",
    ):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _get_bytes(f"{base_url}{blocked}")
        assert exc_info.value.code in (403, 404)


def test_scenario_injection_offline_step_and_loop(web_server):
    base_url, _ = web_server

    # Create a dedicated instance for test mutations so default stays predictable
    c_status, created = _post_json(f"{base_url}/api/instance/create", {"name": "TestRunner", "plan": "logic"})
    assert c_status == 200
    assert created["ok"] is True
    inst_id = created["id"]

    # Configure offline heuristic mode + zero delay
    cfg_status, cfg_res = _post_json(
        f"{base_url}/api/config",
        {
            "instance_id": inst_id,
            "llm_mode": "offline",
            "steps": 2,
            "delay": 0.0,
        },
    )
    assert cfg_status == 200
    assert cfg_res["ok"] is True

    # Inject a custom battle scenario
    sc_status, sc_res = _post_json(
        f"{base_url}/api/scenario",
        {
            "instance_id": inst_id,
            "plan": "logic",
            "scenario_preset": "battle_turn",
            "battle_overrides": {"stamina": 19, "genki": 5, "score": 880},
        },
    )
    assert sc_status == 200
    assert sc_res["scene_type"] == "battle_turn"

    # Execute 1 decision step offline
    st_status, st_res = _post_json(f"{base_url}/api/step", {"instance_id": inst_id})
    assert st_status == 200
    assert st_res["ok"] is True
    assert st_res["source"] in ("rule_skip", "fallback")
    assert "LLM异常" not in st_res["reason"]
    assert st_res["step"] == 1

    # Start and stop background loop cleanly
    loop_status, loop_res = _post_json(f"{base_url}/api/start", {"instance_id": inst_id})
    assert loop_status == 200
    assert loop_res["ok"] is True

    deadline = time.time() + 4.0
    while time.time() < deadline:
        _, _, raw_s = _get_bytes(f"{base_url}/api/state")
        s = json.loads(raw_s.decode("utf-8"))
        if not s["active"]["running"] and s["active"]["step_count"] >= 2:
            break
        time.sleep(0.05)

    _, _, raw_s = _get_bytes(f"{base_url}/api/state")
    final_s = json.loads(raw_s.decode("utf-8"))
    assert final_s["active"]["id"] == inst_id
    assert final_s["active"]["step_count"] >= 2
    assert len(final_s["active"]["logs"]) >= 2

    # Switch back to default instance
    _post_json(f"{base_url}/api/instance/select", {"instance_id": "default"})


def test_config_validation_and_sequence_progression(web_server):
    base_url, _ = web_server

    # Invalid enums must return HTTP 400 and leave config untouched
    for bad_payload in (
        {"plan": "invalid_plan"},
        {"mode": "invalid_mode"},
        {"scenario_preset": "invalid_preset"},
        {"llm_mode": "invalid_llm"},
    ):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _post_json(f"{base_url}/api/config", bad_payload)
        assert exc_info.value.code == 400

    # Non-numeric limit on library endpoint returns HTTP 400
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _get_bytes(f"{base_url}/api/library/cards?limit=abc")
    assert exc_info.value.code == 400

    # Sequence preset walks produce_schedule -> card_reward -> shop -> popup -> battle_turn
    _post_json(f"{base_url}/api/config", {"llm_mode": "offline", "auto_progress": True})
    sc_status, sc_res = _post_json(f"{base_url}/api/scenario", {"scenario_preset": "sequence", "plan": "sense"})
    assert sc_status == 200
    assert sc_res["scene_type"] == "produce_schedule"

    expected_scenes = ["produce_schedule", "card_reward", "shop", "popup"]
    observed_scenes = []
    for _ in expected_scenes:
        _post_json(f"{base_url}/api/step", {})
        _, _, raw_s = _get_bytes(f"{base_url}/api/state")
        s = json.loads(raw_s.decode("utf-8"))
        observed_scenes.append(s["active"]["observation"]["scene_type"])
    assert observed_scenes == expected_scenes

    # Refresh all mock instances
    p_status, p_res = _post_json(f"{base_url}/api/preview", {"all": True})
    assert p_status == 200
    assert p_res["ok"] is True
    assert "default" in p_res["refreshed"]

