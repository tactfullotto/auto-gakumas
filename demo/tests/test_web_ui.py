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


def test_hif_catalog_and_idol_selection(web_server):
    base_url, _ = web_server

    # Static JS workbench asset served cleanly
    js_status, _, js_bytes = _get_bytes(f"{base_url}/static/hif_workbench.js")
    assert js_status == 200
    assert b"/api/hif/catalog" in js_bytes

    # H.I.F catalog exposes 3 professions, 165 P-Idols, P-items, and memory catalog
    c_status, _, c_raw = _get_bytes(f"{base_url}/api/hif/catalog")
    assert c_status == 200
    cat = json.loads(c_raw.decode("utf-8"))
    assert set(cat["plans"].keys()) == {"sense", "logic", "anomaly"}
    assert len(cat["p_idols"]) == 165
    assert any(p["icon_url"] for p in cat["p_idols"])
    assert len(cat["p_items"]) >= 8
    mem_cat = cat["memory_catalog"]
    assert "logic" in mem_cat["plans"]
    assert len(mem_cat["plans"]["logic"]["extra_use_draw_1"]) > 5

    # Default PNG placeholder served on /api/hif/frame
    f_status, f_headers, f_bytes = _get_bytes(f"{base_url}/api/hif/frame")
    assert f_status == 200
    assert f_headers.get("Content-Type") == "image/png"
    assert f_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    # Select Misuzu (#163 logic) and auto-seed signature cards + unique P-item
    s_status, s_res = _post_json(
        f"{base_url}/api/hif/select-idol",
        {"plan": "logic", "p_idol_id": 163, "auto_seed_signature": True},
    )
    assert s_status == 200
    assert s_res["ok"] is True
    st = s_res["state"]
    assert st["selected_plan"] == "logic"
    assert st["selected_p_idol_id"] == 163
    assert any(cid in st["deck_ids"] for cid in (919, 920, 921, 922))
    assert any(it.get("item_id") == 509 for it in st["p_items"])

    # Switch to Sense idol (#158 Agitato 葛城リーリヤ)
    s2_status, s2_res = _post_json(
        f"{base_url}/api/hif/select-idol",
        {"plan": "sense", "p_idol_id": 158, "auto_seed_signature": True},
    )
    assert s2_status == 200
    assert s2_res["state"]["selected_plan"] == "sense"
    assert 884 in s2_res["state"]["deck_ids"] or 883 in s2_res["state"]["deck_ids"]


def test_hif_manual_loadout_memories_and_preferences_roundtrip(web_server):
    base_url, _ = web_server

    # Update run config + manual loadout mode
    cfg_st, cfg_res = _post_json(
        f"{base_url}/api/hif/config",
        {
            "loadout_mode": "manual",
            "adb_host": "127.0.0.1",
            "adb_port": 16384,
            "scenario_id": "hif_honsen",
            "max_steps": 300,
        },
    )
    assert cfg_st == 200
    assert cfg_res["state"]["loadout_mode"] == "manual"
    assert cfg_res["state"]["run_config"]["adb_port"] == 16384

    # Save manual skill cards, P-items, stats
    lo_st, lo_res = _post_json(
        f"{base_url}/api/hif/loadout",
        {
            "deck_ids": [920, 922, 137, 299],
            "idol_stats": {"vocal": 2550, "dance": 1300, "visual": 2050},
            "p_items": [
                {
                    "item_id": 509,
                    "name": "星々を統べる輝き+",
                    "canonical_name": "星々を統べる輝き+",
                    "mode": "stage",
                    "source_type": "pIdol",
                    "rarity": "SSR",
                }
            ],
        },
    )
    assert lo_st == 200
    assert lo_res["state"]["deck_ids"] == [920, 922, 137, 299]
    assert lo_res["state"]["idol_stats"]["vocal"] == 2550

    # Save manual H.I.F card hooks + 4 memory slots
    mem_st, mem_res = _post_json(
        f"{base_url}/api/hif/memories",
        {
            "card_memory_hooks": {
                "299": {"free_extra_use_first_play": True, "extra_use_limit": 2, "draw_on_play": 1}
            },
            "memory_slots": [
                {
                    "slot": 1,
                    "label": "メモリー #1",
                    "owner_name": "美鈴 S+ メモリー",
                    "passives": [
                        {"key": "start_good_impression", "label": "試験開始時 好印象+X", "value": 4},
                    ],
                },
                {"slot": 2, "label": "メモリー #2", "owner_name": "", "passives": []},
                {"slot": 3, "label": "メモリー #3", "owner_name": "", "passives": []},
                {"slot": 4, "label": "メモリー #4", "owner_name": "", "passives": []},
            ],
        },
    )
    assert mem_st == 200
    assert mem_res["state"]["card_memory_hooks"]["299"]["extra_use_limit"] == 2
    assert mem_res["state"]["card_memory_hooks"]["299"]["draw_on_play"] == 1
    assert mem_res["state"]["memory_passives"][0]["key"] == "start_good_impression"

    # Save preferences (prioritize #171, avoid #152, ban #19 + #2, prefer #25)
    pref_st, pref_res = _post_json(
        f"{base_url}/api/hif/preferences",
        {
            "priority_card_ids": [171],
            "keep_card_ids": [922],
            "avoid_card_ids": [152],
            "card_score_overrides": {"299": 750},
            "preferred_drink_ids": [25],
            "banned_drink_ids": [19, 2],
            "spend_all_final_shop": True,
            "reroll_early_non_exhaust_mental": False,
        },
    )
    assert pref_st == 200
    prefs = pref_res["state"]["preferences"]
    assert prefs["priority_card_ids"] == [171]
    assert prefs["banned_drink_ids"] == [19, 2]
    assert prefs["reroll_early_non_exhaust_mental"] is False

    # Export JSON and re-import via {data}
    ex_st, _, ex_raw = _get_bytes(f"{base_url}/api/hif/export-library")
    assert ex_st == 200
    exported = json.loads(ex_raw.decode("utf-8"))
    assert exported["deck_ids"] == [920, 922, 137, 299]
    assert exported["preferences"]["preferred_drink_ids"] == [25]

    imp_st, imp_res = _post_json(f"{base_url}/api/hif/import-library", {"data": exported})
    assert imp_st == 200
    assert imp_res["state"]["deck_ids"] == [920, 922, 137, 299]
    assert len(imp_res["state"]["memory_passives"]) == 1

    # Regression: importing JSON where memory_slots has passives (+6 好調/+6 集中) and stale memory_passives: []
    # must preserve slot-derived passives rather than overwriting them with [].
    stale_payload = dict(exported)
    stale_payload["memory_slots"] = [
        {
            "slot": 1,
            "label": "メモリー #1",
            "owner_name": "好调+6",
            "passives": [
                {"key": "start_good_condition", "label": "試験開始時 好調+Xターン", "value": 6},
            ],
        },
        {
            "slot": 2,
            "label": "メモリー #2",
            "owner_name": "集中+6",
            "passives": [
                {"key": "start_concentration", "label": "試験開始時 集中+X", "value": 6},
            ],
        },
        {"slot": 3, "label": "メモリー #3", "owner_name": "", "passives": []},
        {"slot": 4, "label": "メモリー #4", "owner_name": "", "passives": []},
    ]
    stale_payload["memory_passives"] = []
    imp2_st, imp2_res = _post_json(f"{base_url}/api/hif/import-library", {"data": stale_payload})
    assert imp2_st == 200
    assert [
        (p["key"], p["value"]) for p in imp2_res["state"]["memory_passives"]
    ] == [("start_good_condition", 6), ("start_concentration", 6)]

    # Standalone memory_passives with empty slot passives should mirror into slot 1 for editing
    standalone_payload = dict(exported)
    standalone_payload["memory_slots"] = [
        {"slot": idx, "label": f"メモリー #{idx}", "owner_name": "", "passives": []}
        for idx in range(1, 5)
    ]
    standalone_payload["memory_passives"] = [
        {"key": "start_good_condition", "label": "試験開始時 好調+Xターン", "value": 6},
        {"key": "start_concentration", "label": "試験開始時 集中+X", "value": 6},
    ]
    imp3_st, imp3_res = _post_json(f"{base_url}/api/hif/import-library", {"data": standalone_payload})
    assert imp3_st == 200
    assert len(imp3_res["state"]["memory_passives"]) == 2
    assert len(imp3_res["state"]["memory_slots"][0]["passives"]) == 2

    # Security: path traversal in import filename must return HTTP 400/403
    for bad_name in ("../manifest.json", "demo/schemas.py", "..\\gakumas_library.db"):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _post_json(f"{base_url}/api/hif/import-library", {"filename": bad_name})
        assert exc_info.value.code in (400, 403)


def test_full_library_card_pool_and_card_customizations_roundtrip(web_server):
    base_url, _ = web_server

    # 1. Full plan card pool without truncation (#299 オトメゴコロ+ must be returned when browsing logic)
    st_logic, _, raw_logic = _get_bytes(f"{base_url}/api/library/cards?plan=logic&limit=1000")
    assert st_logic == 200
    logic_data = json.loads(raw_logic.decode("utf-8"))
    assert logic_data["total"] >= 380
    assert len(logic_data["items"]) == logic_data["total"]
    card_299 = next((c for c in logic_data["items"] if c["id"] == 299), None)
    assert card_299 is not None
    assert "オトメゴコロ" in card_299["name"]
    assert card_299["available_customization_ids"] == [34, 80]

    # Full cross-plan library explorer query
    st_all, _, raw_all = _get_bytes(f"{base_url}/api/library/cards?plan=all&limit=1200")
    assert st_all == 200
    all_data = json.loads(raw_all.decode("utf-8"))
    assert all_data["total"] >= 900
    assert len(all_data["items"]) == all_data["total"]

    # 2. Catalog exposes all 111 official customizations
    st_cat, _, raw_cat = _get_bytes(f"{base_url}/api/hif/catalog")
    assert st_cat == 200
    cat = json.loads(raw_cat.decode("utf-8"))
    cust_cat = cat.get("customizations") or {}
    assert len(cust_cat) == 111
    assert cust_cat["36"]["force_initial_hand"] is True
    assert cust_cat["55"]["max_level"] == 2

    # 3. Save per-card official customizations + generic modifiers & roundtrip via export/import
    lo_st, lo_res = _post_json(
        f"{base_url}/api/hif/loadout",
        {
            "deck_ids": [127, 299, 920],
            "card_customizations": {
                "127": {"levels": {"36": 1, "55": 2}},
                "299": {"levels": {"34": 1, "80": 1}, "modifiers": {"score_delta": 15}},
                "920": {"modifiers": {"force_initial_hand": True, "good_impression_delta": 3}},
            },
        },
    )
    assert lo_st == 200
    saved_custs = lo_res["state"]["card_customizations"]
    assert saved_custs["127"]["levels"] == {"36": 1, "55": 2}
    assert saved_custs["299"]["levels"] == {"34": 1, "80": 1}
    assert saved_custs["299"]["modifiers"]["score_delta"] == 15
    assert saved_custs["920"]["modifiers"]["force_initial_hand"] is True

    deck_127 = next(c for c in lo_res["state"]["deck_cards"] if c["card_id"] == 127)
    assert deck_127["available_customization_ids"] == [36, 29, 55]
    assert deck_127["customizations"]["levels"] == {"36": 1, "55": 2}

    ex_st, _, ex_raw = _get_bytes(f"{base_url}/api/hif/export-library")
    assert ex_st == 200
    exported = json.loads(ex_raw.decode("utf-8"))
    assert exported["card_customizations"]["299"]["modifiers"]["score_delta"] == 15

    # Clear then re-import
    _post_json(f"{base_url}/api/hif/loadout", {"card_customizations": {}})
    imp_st, imp_res = _post_json(f"{base_url}/api/hif/import-library", {"data": exported})
    assert imp_st == 200
    assert imp_res["state"]["card_customizations"]["127"]["levels"] == {"36": 1, "55": 2}
    assert imp_res["state"]["card_customizations"]["299"]["modifiers"]["score_delta"] == 15

    # 4. Selecting 莉波 (#139) seeds Vi > Da > Vo default stats; selecting 美鈴 (#163) seeds Vo > Vi > Da
    rin_st, rin_res = _post_json(f"{base_url}/api/hif/select-idol", {"p_idol_id": 139, "auto_seed_signature": True})
    assert rin_st == 200
    r_stats = rin_res["state"]["idol_stats"]
    assert r_stats["visual"] > r_stats["dance"] > r_stats["vocal"]

    mis_st, mis_res = _post_json(f"{base_url}/api/hif/select-idol", {"p_idol_id": 163, "auto_seed_signature": True})
    assert mis_st == 200
    m_stats = mis_res["state"]["idol_stats"]
    assert m_stats["vocal"] > m_stats["visual"] > m_stats["dance"]

    # 5. Selecting #136 花海佑芽 seeds unupgraded 夢を創るステッキ (9136, 110%) by default,
    #    with 91361 (夢を創るステッキ+, 130%) paired in catalog for manual UI toggle
    cat_st, _, cat_raw = _get_bytes(f"{base_url}/api/hif/catalog")
    assert cat_st == 200
    p_items_cat = {int(it["item_id"]): it for it in json.loads(cat_raw.decode("utf-8"))["p_items"]}
    assert 9136 in p_items_cat and 91361 in p_items_cat
    assert p_items_cat[9136]["upgrade_pair_id"] == 91361 and p_items_cat[9136]["upgraded"] is False
    assert p_items_cat[91361]["upgrade_pair_id"] == 9136 and p_items_cat[91361]["upgraded"] is True

    ume_st, ume_res = _post_json(f"{base_url}/api/hif/select-idol", {"p_idol_id": 136, "auto_seed_signature": True})
    assert ume_st == 200
    u_items = ume_res["state"]["p_items"]
    assert len(u_items) >= 1 and int(u_items[0]["item_id"]) == 9136
    assert u_items[0]["canonical_name"] == "夢を創るステッキ"
    assert u_items[0].get("upgrade_pair_id") == 91361
    # Verify loadout save accepts manual switch to 91361 (夢を創るステッキ+) and preserves pair fields
    sw_st, sw_res = _post_json(f"{base_url}/api/hif/loadout", {"p_items": [p_items_cat[91361]]})
    assert sw_st == 200
    assert int(sw_res["state"]["p_items"][0]["item_id"]) == 91361
    assert sw_res["state"]["p_items"][0]["canonical_name"] == "夢を創るステッキ+"
    assert sw_res["state"]["p_items"][0].get("upgraded") is True


def test_unlimited_preferences_and_pref_search_filters(web_server):
    """Preferences must support >15 cards (e.g. 40+ avoid cards) without cap, and index.html exposes multi-filters."""
    base_url, _ = web_server

    # Fetch real Sense card IDs (>40)
    st_s, _, raw_s = _get_bytes(f"{base_url}/api/library/cards?plan=sense&limit=500")
    assert st_s == 200
    sense_ids = [int(c["id"]) for c in json.loads(raw_s.decode("utf-8"))["items"] if (c.get("type") or "") != "trouble"]
    assert len(sense_ids) >= 60

    avoid_40 = sense_ids[:40]
    pri_20 = sense_ids[40:60]
    keep_18 = sense_ids[60:78]
    weights_20 = {str(cid): 200 for cid in sense_ids[:20]}

    p_st, p_res = _post_json(
        f"{base_url}/api/hif/preferences",
        {
            "priority_card_ids": pri_20,
            "keep_card_ids": keep_18,
            "avoid_card_ids": avoid_40,
            "card_score_overrides": weights_20,
            "preferred_drink_ids": [22, 18, 14],
            "banned_drink_ids": [19],
        },
    )
    assert p_st == 200
    prefs = p_res["state"]["preferences"]
    assert len(prefs["avoid_card_ids"]) == 40
    assert len(prefs["priority_card_ids"]) == 20
    assert len(prefs["keep_card_ids"]) == 18
    assert len(prefs["card_score_overrides"]) == 20

    # Verify index.html contains all preference search filter controls
    h_st, _, h_raw = _get_bytes(f"{base_url}/")
    assert h_st == 200
    html = h_raw.decode("utf-8")
    for elem_id in (
        "hif-pref-card-q",
        "hif-pref-card-type",
        "hif-pref-card-cat",
        "hif-pref-card-rarity",
        "hif-pref-card-up",
        "hif-pref-card-effect",
        "hif-pref-card-status",
        "hif-pref-card-sort",
        "btn-hif-pref-bulk-avoid",
        "btn-hif-pref-reset-filter",
    ):
        assert elem_id in html, f"Missing filter control {elem_id} in index.html"





