"""ING-07 / D-13: AFD pre-filter skips routine (no paid call); signal forces tool-use JSON.

GREEN (02-03). The keyword pre-filter (``afd_should_classify``) runs BEFORE any Anthropic
call to stay in budget: routine AFDs ("near normal") are skipped with ZERO client calls;
signal AFDs ("model disagreement") are classified via the SDK forced-``tool_choice`` pattern
whose ``input_schema`` guarantees a ``{disagreement, direction, summary}`` dict (replacing
v3's fragile ``json.loads(text)``). The Anthropic SDK is MOCKED here — no real network, no
paid call. Uses the vendored AFD text fixtures.
"""

from __future__ import annotations

import pathlib
import types
from unittest.mock import MagicMock

from weatherquant.ingest.afd import (
    AFD_MODEL,
    CITY_WFO,
    afd_should_classify,
    classify_afd,
)
from weatherquant.registry import CITIES

_FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def _routine() -> str:
    return (_FIXTURES / "afd_sample_routine.txt").read_text()


def _signal() -> str:
    return (_FIXTURES / "afd_sample_signal.txt").read_text()


def _mock_client_returning(tool_input: dict) -> MagicMock:
    """A mock Anthropic client whose messages.create returns one tool_use block."""
    tool_use_block = types.SimpleNamespace(type="tool_use", input=tool_input)
    message = types.SimpleNamespace(content=[tool_use_block])
    client = MagicMock()
    client.messages.create.return_value = message
    return client


def test_prefilter_skips_routine_classifies_signal():
    should_routine, reason_routine = afd_should_classify(_routine())
    should_signal, reason_signal = afd_should_classify(_signal())
    assert should_routine is False  # routine text → no paid call (D-13)
    assert "routine" in reason_routine
    assert should_signal is True  # signal text → classify
    assert "signal keyword" in reason_signal


def test_routine_text_makes_zero_anthropic_calls():
    # D-13/budget: a routine AFD must NOT touch the client at all.
    client = _mock_client_returning({"disagreement": True, "direction": "warmer", "summary": "x"})
    result = classify_afd(_routine(), wfo="OKX", client=client)
    client.messages.create.assert_not_called()  # the pre-filter short-circuited
    assert result["disagreement"] is False
    assert "reason" in result  # carries the pre-filter reason


def test_signal_text_forces_tool_use_and_parses_structured_dict():
    client = _mock_client_returning(
        {"disagreement": True, "direction": "uncertain", "summary": "Model spread on the front."}
    )
    result = classify_afd(_signal(), wfo="OKX", client=client)

    # The signal text proceeds to exactly one (mocked) Anthropic call.
    client.messages.create.assert_called_once()
    _args, kwargs = client.messages.create.call_args
    # Forced tool_choice → guaranteed structured JSON (Pattern 7, D-13).
    assert kwargs["tool_choice"] == {"type": "tool", "name": "record_afd_signal"}
    assert kwargs["model"] == AFD_MODEL
    assert AFD_MODEL.startswith("claude-haiku-4-5")
    assert kwargs["tools"][0]["name"] == "record_afd_signal"
    schema = kwargs["tools"][0]["input_schema"]
    assert set(schema["required"]) == {"disagreement", "direction", "summary"}

    # The tool_use input dict is returned (no json.loads of a text body).
    assert result == {
        "disagreement": True,
        "direction": "uncertain",
        "summary": "Model spread on the front.",
    }


def test_empty_text_returns_no_signal_without_call():
    client = _mock_client_returning({"disagreement": True, "direction": "warmer", "summary": "y"})
    result = classify_afd("", wfo="OKX", client=client)
    client.messages.create.assert_not_called()
    assert result["disagreement"] is False


def test_graceful_skip_when_api_key_unset(monkeypatch):
    # No client injected and no key in Settings → structured skip, no call, no signal (D-11).
    import weatherquant.ingest.afd as afd_mod

    fake_settings = types.SimpleNamespace(anthropic_api_key=None)
    monkeypatch.setattr(afd_mod, "get_settings", lambda: fake_settings)
    result = classify_afd(_signal(), wfo="OKX")  # signal text → would classify, but no key
    assert result["disagreement"] is False
    assert result.get("reason") == "anthropic_api_key unset"


def test_wfo_map_matches_registry_cities():
    # AFD is per-WFO; the map must cover exactly the 7 registry cities (T-02-11).
    assert set(CITY_WFO) == set(CITIES)
    assert CITY_WFO["NYC"] == "OKX"
    assert CITY_WFO["CHI"] == "LOT"
    assert CITY_WFO["DEN"] == "BOU"


def test_malformed_tool_input_is_coerced_to_required_shape():
    # Defensive: even if the tool input lacks keys, the three required keys are present.
    client = _mock_client_returning({"disagreement": True})  # missing direction/summary
    result = classify_afd(_signal(), wfo="OKX", client=client)
    assert result["disagreement"] is True
    assert result["direction"] == ""
    assert result["summary"] == ""
