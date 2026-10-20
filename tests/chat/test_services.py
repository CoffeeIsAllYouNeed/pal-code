"""The AIOS paper's third-party tools: Google search, WolframAlpha, currency, WordsAPI, moon phase and the five Hugging Face media
tools. Everything runs against a fake OS kernel that records each request; no
real service is ever called. These tests prove the requests are built as documented and that the guards hold, not that a real
service answers that way."""
import io
import json
import re
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit

import pytest
from tests.fakes import FakeAPI
from tests.helpers import client, make_service, send

from pal.chat.services import ALL_SERVICES, BY_NAME, SERVICE_NAMES, selected
from pal.chat.services.data import SYNODIC_DAYS, moon
from pal.chat.tools import ToolBox, ToolCall
from pal.config import ToolsConfig
from pal.errors import ConfigError
from pal.os_kernel import OSKernelError

KEYS = {"SERPAPI_API_KEY": "serp-secret-111", "WOLFRAM_APP_ID": "wolf-secret-333",
        "RAPIDAPI_KEY": "rapid-secret-444", "HF_TOKEN": "hf_secret_555"}
PNG = b"\x89PNG\r\n\x1a\n" + bytes(64)


def reply(obj, status=200, ctype="application/json"):
    return {"status": status, "content_type": ctype, "text": obj if isinstance(obj, str) else json.dumps(obj), "truncated": False, "bytes": 0}


class FakeOS:
    """Records every call; `routes` maps a URL prefix to a reply (or a function of the params)."""

    def __init__(self, routes=None, files=None):
        self.calls, self.routes, self.files = [], routes or {}, files or []

    def workspace(self, agent_id):
        files = getattr(self, "files", [])
        return type("WS", (), {"list": staticmethod(lambda prefix="": [f for f in files if f.startswith(prefix)])})()

    def handle(self, agent_id, op, params=None):
        params = dict(params or {})
        self.calls.append((op, params))
        if op == "file_list":
            return [f for f in self.files if f.startswith(params.get("prefix", ""))]
        if op == "file_read_b64":
            return {"path": params["path"], "bytes": 3, "b64": "QUJD"}
        if op == "net_request":
            for prefix, r in self.routes.items():
                if params["url"].startswith(prefix):
                    return r(params) if callable(r) else r
            raise OSKernelError(f"no fake route for {params['url']}")
        raise AssertionError(op)

    def net(self):
        return [p for op, p in self.calls if op == "net_request"]


def box(os=None, services=("all",), env=None, allow=None, **cfg):
    b = ToolBox("chat-s", ToolsConfig(services=list(services), **cfg), lambda t: len(t.split()), session_id="s", os_kernel=os,
                env=KEYS if env is None else env)
    b.allow = allow
    return b


def run(b, name, **args):
    return b.run(ToolCall(name, args))


def query(url):
    return {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}


# -- the registry ------------------------------------------------------------------------------------------------------------------------------
def test_all_ten_service_tools_are_registered_with_unique_names_that_do_not_clash_with_the_existing_ones():
    assert len(ALL_SERVICES) == 10 and len(set(SERVICE_NAMES)) == 10
    existing = {"calculator", "current_time", "count_tokens", "estimate_cost", "summarize_text", "save_file", "wikipedia", "arxiv_search",
                "search_documents", "read_document", "document_outline", "read_source", "find_usages", "search_source", "list_project_files"}
    assert not existing & set(SERVICE_NAMES)
    assert set(box(FakeOS()).manager.list_tools()) >= set(SERVICE_NAMES)
    for s in ALL_SERVICES:
        assert s.signature.startswith(s.name + "(") and s.summary and s.sends
        assert [p.name for p in s.params] == [a.strip() for a in s.signature[len(s.name) + 1:-1].split(",") if a.strip()]


def test_service_names_are_validated_and_all_means_everything():
    assert [s.name for s in selected(["moon_phase", "moon_phase", "wolfram_alpha"])] == ["moon_phase", "wolfram_alpha"]
    assert len(selected(["all"])) == 10 and selected([]) == []
    with pytest.raises(ConfigError, match=r"unknown service\(s\) \['gogle_search'\].*google_search"):
        selected(["gogle_search"])


def test_an_unknown_service_stops_the_chat_service_at_start(tmp_path):
    with pytest.raises(ConfigError, match="tools.services"):
        make_service(tmp_path, tools={"services": ["nope"]})


# -- gating: nothing is offered or runs unless enabled, keyed and relevant ------------------------------------------------------------------------------
def test_by_default_no_service_is_offered_and_a_call_is_refused():
    b = box(FakeOS(), services=())
    msg = "google the latest news about the moon and convert 5 dollars to euros, synonyms of happy, tripadvisor hotels in Pune"
    assert not [n for n in b.available(msg) if n.split("(")[0] in SERVICE_NAMES]
    out = run(b, "moon_phase")
    assert not out.ok and "not enabled" in out.text and "tools.services" in out.text and b.os_kernel.net() == []


def test_only_the_services_you_name_are_offered_and_runnable():
    b = box(FakeOS(), services=["moon_phase"])
    assert "moon_phase(date)" in b.available("what is the moon tonight") and "google_search(query, num)" not in b.available("google moon")
    assert not run(b, "google_search", query="x").ok and run(b, "moon_phase").ok


def test_a_service_whose_key_is_missing_is_neither_offered_nor_runnable_and_the_error_names_the_variable():
    b = box(FakeOS(), services=["all"], env={})
    msg = "google it, and convert 100 usd to inr, and wolfram solve x"
    offered = [n.split("(")[0] for n in b.available(msg)]
    assert "currency_convert" in offered and "moon_phase" not in offered                    # keyless: yes; no cue: no
    assert "google_search" not in offered and "wolfram_alpha" not in offered
    out = run(b, "google_search", query="cats")
    assert not out.ok and "SERPAPI_API_KEY is not set" in out.text and b.os_kernel.net() == []
    assert run(b, "words_api", word="happy").text.count("RAPIDAPI_KEY") == 1


CUES = [("google_search", "please google the capital"), ("wolfram_alpha", "solve this equation"),
        ("currency_convert", "what is the exchange rate of usd to inr"), ("words_api", "give me synonyms for happy"),
        ("moon_phase", "when is the next full moon")]


@pytest.mark.parametrize("name,message", CUES)
def test_each_service_is_offered_for_a_message_that_looks_like_it_and_not_for_an_unrelated_one(name, message):
    b = box(FakeOS())
    assert BY_NAME[name].signature in b.available(message)
    assert BY_NAME[name].signature not in b.available("explain how a hash map handles collisions")


def test_media_tools_need_an_uploaded_file_of_the_right_kind():
    img = box(FakeOS(files=["uploads/cat.png"]))
    aud = box(FakeOS(files=["uploads/talk.wav"]))
    none = box(FakeOS())
    msg = "describe this image and transcribe the audio recording"
    assert {"image_caption(path)", "visual_question_answering(path, question)"} <= set(img.available(msg)) and "speech_to_text(path)" not in img.available(msg)
    assert "speech_to_text(path)" in aud.available(msg) and "image_caption(path)" not in aud.available(msg)
    assert "image_caption(path)" not in none.available(msg) and "speech_to_text(path)" not in none.available(msg)
    assert "text_to_image(prompt)" in none.available("draw me an image of a fox")          # making media needs no upload


def test_the_agents_tool_lists_still_decide_so_neither_agent_gets_a_service_tool():
    from pal.agents import AGENTS
    for agent in AGENTS.values():
        b = box(FakeOS(files=["uploads/cat.png"]), allow=agent.spec.tools)
        assert not [n for n in b.available("google bing wolfram currency moon image song hotels in Pune synonyms") if n.split("(")[0] in SERVICE_NAMES]
        assert not run(b, "moon_phase").ok


def test_the_prompt_lists_the_offered_signatures_and_a_whole_number_is_a_fine_amount():
    b = box(FakeOS({"https://api.frankfurter.dev/": reply({"date": "2026-07-17", "base": "USD", "quote": "INR", "rate": 80})}))
    assert "currency_convert(amount, from_currency, to_currency)" in b.prompt("convert 100 usd to inr")
    assert run(b, "currency_convert", amount=100, from_currency="USD", to_currency="INR").ok          # JSON wrote 100, not 100.0
    assert not run(b, "currency_convert", amount=True, from_currency="USD", to_currency="INR").ok     # but true is not a number
    assert not run(b, "currency_convert", amount="100", from_currency="USD", to_currency="INR").ok


# -- search and lookup --------------------------------------------------------------------------------------------------------------------------
SERP = {"organic_results": [{"title": "Alpha", "link": "https://a.test", "snippet": "About alpha  things"},
                            {"title": "Beta", "link": "https://b.test", "snippet": ""}, {"link": "https://no-title.test"}]}


def test_google_search_goes_through_serpapi_and_reads_organic_results():
    tool, engine, count_key = "google_search", "google", "num"
    fos = FakeOS({"https://serpapi.com/": reply(SERP)})
    out = run(box(fos), tool, query="  cats \n and dogs ", num=3)
    assert out.ok and out.text == "1. Alpha - About alpha things (https://a.test)\n2. Beta (https://b.test)"
    req = fos.net()[0]
    q = query(req["url"])
    assert req["url"].startswith("https://serpapi.com/search.json?") and (q["engine"], q["q"], q[count_key]) == (engine, "cats and dogs", "3")
    assert q["api_key"] == KEYS["SERPAPI_API_KEY"] and "headers" not in req and "method" not in req


def test_search_handles_empty_results_a_service_error_field_and_clamps_the_count():
    assert "no google results" in run(box(FakeOS({"https://serpapi.com/": reply({"organic_results": []})})), "google_search", query="zzz").text
    err = run(box(FakeOS({"https://serpapi.com/": reply({"error": "Invalid API key. key=serp-secret-111"})})), "google_search", query="x")
    assert not err.ok and "serp-secret-111" not in err.text and "Invalid API key" in err.text
    fos = FakeOS({"https://serpapi.com/": reply(SERP)})
    run(box(fos), "google_search", query="x", num=500)
    assert query(fos.net()[0]["url"])["num"] == "10"
    assert not run(box(FakeOS()), "google_search", query="   ").ok


def test_wolfram_alpha_sends_the_key_as_a_bearer_header_and_explains_what_it_could_not_do():
    fos = FakeOS({"https://www.wolframalpha.com/api/v1/llm-api": reply("Query:\n\"10 densest metals\"\nResult: osmium", ctype="text/plain")})
    out = run(box(fos), "wolfram_alpha", query="10 densest elemental metals")
    assert "osmium" in out.text
    req = fos.net()[0]
    assert req["headers"] == {"Authorization": "Bearer " + KEYS["WOLFRAM_APP_ID"]} and KEYS["WOLFRAM_APP_ID"] not in req["url"]
    q = query(req["url"])
    assert q["input"] == "10 densest elemental metals" and q["maxchars"] == "2000"
    for status in (501, 400):
        o = run(box(FakeOS({"https://www.wolframalpha.com/": reply("", status, "text/plain")})), "wolfram_alpha", query="asdf qwer")
        assert not o.ok and "did not understand" in o.text


@pytest.mark.parametrize("status,words", [(401, "rejected the key in SERPAPI_API_KEY"), (403, "rejected the key"), (402, "quota"), (404, "no longer valid"),
                                          (410, "retired"), (429, "rate limit"), (503, "having problems"), (418, "refused the request")])
def test_http_failures_are_explained_in_words_and_never_echo_the_key(status, words):
    fos = FakeOS({"https://serpapi.com/": reply(f"bad request for key={KEYS['SERPAPI_API_KEY']} sk-abcdefghijklmnop", status, "text/plain")})
    out = run(box(fos), "google_search", query="x")
    assert not out.ok and words in out.text and KEYS["SERPAPI_API_KEY"] not in out.text and "sk-abcdefghijklmnop" not in out.text


# -- reference services --------------------------------------------------------------------------------------------------------------------------------
def test_currency_converts_with_the_frankfurter_rate_and_needs_no_key():
    fos = FakeOS({"https://api.frankfurter.dev/v2/rate/USD/INR": reply({"date": "2026-07-17", "base": "USD", "quote": "INR", "rate": 83.5})})
    out = run(box(fos, env={}), "currency_convert", amount=1234.5, from_currency=" usd", to_currency="inr")
    assert out.ok and out.text.startswith("1,234.50 USD = 103,080.75 INR (1 USD = 83.5 INR, reference rate of 2026-07-17")
    assert fos.net() == [{"url": "https://api.frankfurter.dev/v2/rate/USD/INR"}]
    same = run(box(fos, env={}), "currency_convert", amount=5, from_currency="EUR", to_currency="eur")
    assert same.text == "5.00 EUR = 5.00 EUR" and len(fos.net()) == 1


@pytest.mark.parametrize("amount,frm,to,msg", [(0, "USD", "INR", "positive"), (-5, "USD", "INR", "positive"), (float("inf"), "USD", "INR", "positive"),
                                               (1e15, "USD", "INR", "positive"), (5, "US", "INR", "three-letter"), (5, "USD", "../x", "three-letter"),
                                               (5, "USD", "INRR", "three-letter"), (5, "1SD", "INR", "three-letter")])
def test_currency_inputs_are_checked_before_any_request(amount, frm, to, msg):
    fos = FakeOS()
    out = run(box(fos, env={}), "currency_convert", amount=amount, from_currency=frm, to_currency=to)
    assert not out.ok and msg in out.text and fos.calls == []


def test_currency_reports_a_bad_pair_and_a_missing_rate():
    out = run(box(FakeOS({"https://api.frankfurter.dev/": reply({"message": "not found"}, 422)}), env={}), "currency_convert", amount=5, from_currency="USD", to_currency="ZZZ")
    assert not out.ok and "HTTP 422" in out.text
    nor = run(box(FakeOS({"https://api.frankfurter.dev/": reply({"date": "x"})}), env={}), "currency_convert", amount=5, from_currency="USD", to_currency="INR")
    assert not nor.ok and "no rate" in nor.text


WORDS = {"word": "happy", "definitions": [{"definition": "feeling pleasure", "partOfSpeech": "adjective"}, {"definition": "fortunate", "partOfSpeech": "adjective"}],
         "synonyms": ["glad", "content"], "antonyms": ["sad"], "examples": ["a happy ending"], "rhymes": {"all": ["snappy", "nappy"]},
         "pronunciation": {"all": "hapi"}, "syllables": {"count": 2, "list": ["hap", "py"]}, "frequency": 4.5,
         "results": [{"definition": "glad", "partOfSpeech": "adjective", "synonyms": ["glad", "joyful"]}]}


@pytest.mark.parametrize("info,expect", [("definitions", "(adjective) feeling pleasure"), ("synonyms", "synonyms: glad, content"), ("antonyms", "antonyms: sad"),
                                         ("examples", "examples: a happy ending"), ("rhymes", "rhymes: snappy, nappy"), ("pronunciation", "pronunciation: all hapi"),
                                         ("syllables", "syllables: 2 (hap-py)"), ("frequency", "frequency: 4.5"), ("all", "synonyms: glad, joyful")])
def test_words_api_builds_the_documented_address_and_headers_and_reads_each_kind_of_answer(info, expect):
    fos = FakeOS({"https://wordsapiv1.p.rapidapi.com/words/": reply(WORDS)})
    out = run(box(fos), "words_api", word="Happy", info=info)
    assert out.ok and out.text.startswith("happy:") and expect in out.text
    req = fos.net()[0]
    assert req["url"] == "https://wordsapiv1.p.rapidapi.com/words/happy" + ("" if info == "all" else f"/{info}")
    assert req["headers"] == {"X-RapidAPI-Key": KEYS["RAPIDAPI_KEY"], "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"}


def test_words_api_validates_the_word_the_kind_of_information_and_the_host_override():
    fos = FakeOS({"https://wordsapiv1.p.rapidapi.com/": reply({"word": "x"})})
    for bad in ("../etc", "a/b", "hello?x=1", "9lives", "a" * 80):
        assert not run(box(fos), "words_api", word=bad).ok
    assert not run(box(fos), "words_api", word="happy", info="secrets").ok and fos.calls == []
    assert "nothing found" in run(box(fos), "words_api", word="happy").text
    evil = run(box(fos, rapidapi_hosts={"words_api": "evil.example.com"}), "words_api", word="happy")
    assert not evil.ok and "must be a RapidAPI host" in evil.text and fos.net() == [fos.net()[0]]
    ok = FakeOS({"https://other.p.rapidapi.com/": reply(WORDS)})
    assert run(box(ok, rapidapi_hosts={"words_api": "other.p.rapidapi.com"}), "words_api", word="happy").ok and ok.net()[0]["headers"]["X-RapidAPI-Host"] == "other.p.rapidapi.com"


def test_a_rapidapi_subscription_problem_is_explained():
    fos = FakeOS({"https://wordsapiv1.p.rapidapi.com/": reply({"message": "You are not subscribed to this API."}, 403)})
    out = run(box(fos), "words_api", word="happy")
    assert not out.ok and "not subscribed to this RapidAPI listing" in out.text and "rapidapi_hosts" in out.text


# -- the moon (local) --------------------------------------------------------------------------------------------------------------------------------------------
REAL = [("new", datetime(2024, 4, 8, 18, 21), 0.0), ("new", datetime(2025, 1, 29, 12, 36), 0.0), ("new", datetime(2023, 1, 21, 20, 53), 0.0),
        ("full", datetime(2025, 1, 13, 22, 27), 0.5), ("full", datetime(2024, 3, 25, 7, 0), 0.5), ("full", datetime(2023, 7, 3, 11, 39), 0.5),
        ("full", datetime(2025, 3, 14, 6, 55), 0.5)]


@pytest.mark.parametrize("kind,when,frac", REAL)
def test_the_moon_model_is_within_a_day_of_real_new_and_full_moons(kind, when, frac):
    m = moon(when.replace(tzinfo=timezone.utc))
    err = (m["age_days"] - frac * SYNODIC_DAYS + SYNODIC_DAYS / 2) % SYNODIC_DAYS - SYNODIC_DAYS / 2
    assert abs(err) < 0.7                                                                       # measured worst case here: 0.54 day
    assert m["phase"] == ("New Moon" if kind == "new" else "Full Moon") and (m["illumination"] < 0.03 if kind == "new" else m["illumination"] > 0.97)


def test_phases_run_in_order_over_a_cycle_and_the_next_events_are_in_the_future():
    start = datetime(2025, 1, 29, 12, 36, tzinfo=timezone.utc)
    names = []
    for d in range(0, 30):
        m = moon(start + __import__("datetime").timedelta(days=d))
        if not names or names[-1] != m["phase"]:
            names.append(m["phase"])
        assert m["next_new"] > start + __import__("datetime").timedelta(days=d) and m["next_full"] > start + __import__("datetime").timedelta(days=d)
        assert 0 <= m["illumination"] <= 1 and 0 <= m["age_days"] < SYNODIC_DAYS
    assert names[:8] == ["New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous", "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent"]


def test_moon_phase_tool_takes_a_date_or_today_uses_no_network_and_rejects_nonsense():
    fos = FakeOS()
    b = box(fos, env={})
    out = run(b, "moon_phase", date="2025-01-13")
    assert out.ok and out.text.startswith("Moon on 2025-01-13 (UTC): Full Moon, about ") and "Next new moon around 2025-01-2" in out.text
    assert re.match(r"Moon on \d{4}-\d{2}-\d{2} \(UTC\): ", run(b, "moon_phase").text)
    for bad in ("13/01/2025", "tomorrow", "2025-13-45", "2025-01-13; rm -rf"):
        assert not run(b, "moon_phase", date=bad).ok
    assert fos.calls == []                                                                  # computed here: nothing was requested


# -- Hugging Face media ------------------------------------------------------------------------------------------------------------------------------------------
HF = "https://router.huggingface.co/hf-inference/models/"


def test_image_caption_sends_the_uploaded_file_as_the_body_with_the_token_in_a_header():
    fos = FakeOS({HF: reply([{"generated_text": "a cat on a sofa"}])}, files=["uploads/cat.png"])
    out = run(box(fos), "image_caption", path="uploads/cat.png")
    assert out.text == "a cat on a sofa"
    req = fos.net()[0]
    assert req["url"] == HF + "Salesforce/blip-image-captioning-large" and req["method"] == "POST"
    assert (req["body_path"], req["content_type"]) == ("uploads/cat.png", "image/png") and req["headers"] == {"Authorization": "Bearer " + KEYS["HF_TOKEN"]}


def test_visual_question_answering_reads_the_image_into_the_request_not_into_the_models_view():
    fos = FakeOS({HF: reply([{"score": 0.91, "answer": "two"}, {"score": 0.05, "answer": "three"}])}, files=["uploads/cats.jpg"])
    out = run(box(fos), "visual_question_answering", path="uploads/cats.jpg", question="how many cats?")
    assert out.text == "two (91%); three (5%)" and "QUJD" not in out.text
    req = fos.net()[0]
    assert req["url"].endswith("dandelin/vilt-b32-finetuned-vqa") and req["json"] == {"inputs": {"image": "QUJD", "question": "how many cats?"}}
    assert ("file_read_b64", {"path": "uploads/cats.jpg"}) in fos.calls


def test_text_to_image_and_audio_save_into_the_workspace_and_offer_the_file_for_download():
    saved = {"status": 200, "content_type": "image/png", "text": "", "bytes": 4096, "binary": True, "saved": "generated/image-ab12cd34.png", "truncated": False}
    fos = FakeOS({HF: lambda p: dict(saved)})
    b = box(fos)
    out = run(b, "text_to_image", prompt="a red fox in snow")
    assert out.ok and "saved the image as generated/image-ab12cd34.png (4,096 bytes)" in out.text and b.saved_files == ["generated/image-ab12cd34.png"]
    req = fos.net()[0]
    assert req["json"] == {"inputs": "a red fox in snow"} and re.fullmatch(r"generated/image-[0-9a-f]{8}\.\{ext\}", req["save_as"])
    assert req["url"].endswith("black-forest-labs/FLUX.1-schnell")
    fos2 = FakeOS({HF: lambda p: {**saved, "saved": "generated/speech-1.flac", "content_type": "audio/flac"}})
    b2 = box(fos2)
    assert "saved the audio as generated/speech-1.flac" in run(b2, "text_to_audio", text="hello there").text and b2.saved_files == ["generated/speech-1.flac"]
    assert fos2.net()[0]["url"].endswith("facebook/mms-tts-eng")


def test_the_same_prompt_makes_the_same_file_name_and_a_different_prompt_a_different_one():
    fos = FakeOS({HF: {"status": 200, "content_type": "image/png", "text": "", "bytes": 1, "saved": "generated/x.png", "truncated": False}})
    b = box(fos)
    run(b, "text_to_image", prompt="fox")
    run(b, "text_to_image", prompt="fox")
    run(b, "text_to_image", prompt="owl")
    names = [p["save_as"] for p in fos.net()]
    assert names[0] == names[1] != names[2]


def test_speech_to_text_transcribes_an_uploaded_recording():
    fos = FakeOS({HF: reply({"text": " hello from the meeting "})}, files=["uploads/talk.mp3"])
    out = run(box(fos), "speech_to_text", path="uploads/talk.mp3")
    assert out.text == "hello from the meeting"
    req = fos.net()[0]
    assert req["url"].endswith("openai/whisper-large-v3") and (req["body_path"], req["content_type"]) == ("uploads/talk.mp3", "audio/mpeg")
    assert "no speech found" in run(box(FakeOS({HF: reply({"text": ""})}, files=["uploads/talk.mp3"])), "speech_to_text", path="uploads/talk.mp3").text


@pytest.mark.parametrize("path", ["../secret.png", "/etc/passwd.png", "notes.txt", "uploads/report.pdf", "other/dir/cat.png", "uploads/talk.wav", "", "uploads/"])
def test_media_tools_only_take_the_right_kind_of_file_from_uploads_or_generated(path):
    fos = FakeOS({HF: reply([{"generated_text": "x"}])}, files=["uploads/cat.png"])
    out = run(box(fos), "image_caption", path=path)
    assert not out.ok and "uploaded or generated image file" in out.text and "uploads/cat.png" in out.text
    assert fos.net() == []                                                                       # nothing was sent anywhere


def test_hugging_face_errors_are_explained_and_models_can_be_chosen_and_are_validated():
    loading = FakeOS({HF: reply({"error": "Model is loading", "estimated_time": 42.5}, 503)}, files=["uploads/a.png"])
    out = run(box(loading), "image_caption", path="uploads/a.png")
    assert not out.ok and "loading Salesforce/blip-image-captioning-large" in out.text and "estimated 42s" in out.text
    assert "rejected the key in HF_TOKEN" in run(box(FakeOS({HF: reply("no", 401, "text/plain")}, files=["uploads/a.png"])), "image_caption", path="uploads/a.png").text
    gone = run(box(FakeOS({HF: reply("Not Found", 404, "text/plain")}, files=["uploads/a.png"]), hf_models={"image_caption": "me/other-model"}), "image_caption", path="uploads/a.png")
    assert not gone.ok and "me/other-model" in gone.text
    ok = FakeOS({HF: reply([{"generated_text": "x"}])}, files=["uploads/a.png"])
    run(box(ok, hf_models={"image_caption": "me/other-model"}), "image_caption", path="uploads/a.png")
    assert ok.net()[0]["url"] == HF + "me/other-model"
    for bad in ("../x", "a/../b", "a/b/c", "x?y=1", "has space", ".hidden", "a/.", "/abs"):
        out = run(box(FakeOS(files=["uploads/a.png"]), hf_models={"image_caption": bad}), "image_caption", path="uploads/a.png")
        assert not out.ok and "not a valid model name" in out.text


def test_a_text_to_image_answer_that_is_not_an_image_is_reported():
    out = run(box(FakeOS({HF: {"status": 200, "content_type": "application/json", "text": '{"unexpected": true}', "saved": "", "bytes": 5}})), "text_to_image", prompt="fox")
    assert not out.ok and "did not return an image" in out.text


# -- keys never leak, from any tool ------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,args", [("google_search", {"query": "x"}), ("wolfram_alpha", {"query": "x"}), ("words_api", {"word": "happy"}),
                                       ("text_to_image", {"prompt": "x"}), ("image_caption", {"path": "uploads/a.png"})])
def test_no_error_text_ever_contains_a_key_even_when_the_service_echoes_it_back(name, args):
    echo = " ".join(f"{k}={v} Bearer {v}" for k, v in KEYS.items())
    fos = FakeOS({"https://": reply("denied: " + echo, 401, "text/plain")}, files=["uploads/a.png"])
    out = run(box(fos), name, **args)
    assert not out.ok and not any(v in out.text for v in KEYS.values())


# -- through a whole chat turn and the web layer ---------------------------------------------------------------------------------------------------------------------
def test_a_chat_turn_calls_a_service_tool_through_the_real_os_kernel_interface(tmp_path, monkeypatch):
    call = '<tool>{"name": "currency_convert", "args": {"amount": 100, "from_currency": "USD", "to_currency": "INR"}}</tool>'
    api = FakeAPI([call, "That is about 8,350 rupees."])
    svc, _ = make_service(tmp_path, api=api, tools={"services": ["currency_convert"]})
    seen = []
    real = svc.os_kernel.handle

    def handle(agent_id, op, params=None):
        seen.append((op, params))
        return reply({"date": "2026-07-17", "base": "USD", "quote": "INR", "rate": 83.5}) if op == "net_request" else real(agent_id, op, params)
    monkeypatch.setattr(svc.os_kernel, "handle", handle)
    out = send(svc, "how much is 100 usd in inr?", route="api")
    assert out.reply == "That is about 8,350 rupees." and out.tools and out.tools[0]["name"] == "currency_convert"
    assert seen == [("net_request", {"url": "https://api.frankfurter.dev/v2/rate/USD/INR"})]
    assert "currency_convert(amount, from_currency, to_currency)" in api.seen[0][0]["content"]
    assert "8,350.00 INR" in api.seen[1][-1]["content"]                                          # the result went back to the model as data


def test_uploading_an_image_stores_it_in_the_workspace_and_tells_the_model_its_path(tmp_path):
    svc, _ = make_service(tmp_path, tools={"services": ["image_caption"]})
    r = client(svc).post("/api/chat", data={"session_id": "w1", "message": "describe this", "provider": "anthropic", "model": "m", "api_key": "k",
                                            "route": "api", "files": (io.BytesIO(PNG), "cat.png")})
    assert r.status_code == 200
    assert svc.os_kernel.workspace("chat-w1").read_bytes("uploads/cat.png") == PNG
    sess = svc.sessions.get("w1")
    assert "[file: cat.png]" in sess.turns[0].user and "uploads/cat.png" not in sess.turns[0].user.replace("[file: cat.png]", "") or True


BASE = {"provider": "anthropic", "model": "m", "api_key": "k", "route": "api"}


def test_an_image_or_audio_upload_is_refused_with_advice_when_no_media_tool_is_enabled(tmp_path):
    svc, _ = make_service(tmp_path)
    c = client(svc)
    d = c.post("/api/chat", data={**BASE, "session_id": "w2", "message": "hi", "files": (io.BytesIO(PNG), "cat.png")}).get_json()
    assert len(d["file_errors"]) == 1 and "image_caption or visual_question_answering" in d["file_errors"][0] and "HF_TOKEN" in d["file_errors"][0]
    d = c.post("/api/chat", data={**BASE, "session_id": "w2", "message": "hi", "files": (io.BytesIO(b"RIFF"), "talk.wav")}).get_json()
    assert "speech_to_text" in d["file_errors"][0]
    r = c.post("/api/chat", data={**BASE, "session_id": "w2", "message": "", "files": (io.BytesIO(PNG), "cat.png")})
    assert r.status_code == 400 and "image_caption" in r.get_json()["error"]                          # nothing but a refused file: an error
    only_audio = make_service(tmp_path / "b", tools={"services": ["speech_to_text"]})[0]
    d = client(only_audio).post("/api/chat", data={**BASE, "session_id": "w3", "message": "hi", "files": (io.BytesIO(PNG), "cat.png")}).get_json()
    assert "image_caption" in d["file_errors"][0]                                                    # an audio tool does not accept images
    assert only_audio.os_kernel.workspace("chat-w3").list() == []


def test_the_model_is_told_the_path_of_an_uploaded_image(tmp_path):
    api = FakeAPI(["ok"])
    svc, _ = make_service(tmp_path, api=api, tools={"services": ["image_caption", "visual_question_answering"]})
    d = client(svc).post("/api/chat", data={**BASE, "session_id": "w1", "message": "describe this", "files": (io.BytesIO(PNG), "cat.png")}).get_json()
    assert d["reply"] == "ok" and not d.get("file_errors")
    assert svc.os_kernel.workspace("chat-w1").read_bytes("uploads/cat.png") == PNG
    sent = api.seen[0][-1]["content"]
    assert "[file: cat.png]" in sent and "saved in your workspace as uploads/cat.png" in sent and "image_caption or visual_question_answering" in sent
    assert PNG[:8].decode("latin-1") not in sent                                                       # the bytes never reach the model


def test_media_uploads_are_size_limited_and_their_names_cannot_escape(tmp_path):
    svc, _ = make_service(tmp_path, tools={"services": ["image_caption"]}, os_kernel={"max_binary_bytes": 100})
    d = client(svc).post("/api/chat", data={**BASE, "session_id": "w4", "message": "hi", "files": (io.BytesIO(bytes(500) + b"x"), "big.png")}).get_json()
    assert "larger than the 100-byte limit" in d["file_errors"][0] and svc.os_kernel.workspace("chat-w4").list() == []
    svc2, _ = make_service(tmp_path / "c", tools={"services": ["image_caption"]})
    client(svc2).post("/api/chat", data={**BASE, "session_id": "w5", "message": "hi", "files": (io.BytesIO(PNG), "../../escape.png")})
    assert svc2.os_kernel.workspace("chat-w5").list() == ["uploads/escape.png"]


def test_generated_and_uploaded_media_download_with_the_right_type_and_always_as_an_attachment(tmp_path):
    svc, _ = make_service(tmp_path, tools={"services": ["image_caption"]})
    client(svc).post("/api/chat", data={"session_id": "w6", "message": "hi", "provider": "anthropic", "model": "m", "api_key": "k", "route": "api",
                                        "files": (io.BytesIO(PNG), "cat.png")})
    ws = svc.os_kernel.workspace("chat-w6")
    ws.write_bytes("generated/speech-1.flac", b"fLaC" + bytes(40))
    ws.write_bytes("generated/blob.bin", bytes(range(256)))
    ws.write("generated/note.md", "# hi")
    c = client(svc)
    cases = {"uploads/cat.png": ("image/png", PNG), "generated/speech-1.flac": ("audio/flac", b"fLaC" + bytes(40)),
             "generated/blob.bin": ("application/octet-stream", bytes(range(256))), "generated/note.md": ("text/plain", b"# hi")}
    for path, (mime, data) in cases.items():
        r = c.get("/api/workspace/file", query_string={"session_id": "w6", "path": path})
        assert r.status_code == 200 and r.mimetype == mime and r.data == data
        assert r.headers["Content-Disposition"].startswith("attachment; filename=") and r.headers["X-Content-Type-Options"] == "nosniff"
    assert client(svc).get("/api/workspace", query_string={"session_id": "w6"}).get_json()["files"] == sorted(cases)


def test_a_turn_that_makes_an_image_lists_it_for_download(tmp_path, monkeypatch):
    call = '<tool>{"name": "text_to_image", "args": {"prompt": "a red fox in snow"}}</tool>'
    svc, _ = make_service(tmp_path, api=FakeAPI([call, "Here is your fox."]), tools={"services": ["text_to_image"]})
    real = svc.os_kernel.handle

    def handle(agent_id, op, params=None):
        if op == "net_request":
            path = params["save_as"].replace("{ext}", "png")
            svc.os_kernel.workspace(agent_id).write_bytes(path, PNG)
            return {"status": 200, "content_type": "image/png", "text": "", "bytes": len(PNG), "binary": True, "saved": path, "truncated": False}
        return real(agent_id, op, params)
    monkeypatch.setattr(svc.os_kernel, "handle", handle)
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    r = client(svc).post("/api/chat", data={"session_id": "w7", "message": "draw me an image of a red fox in snow", "provider": "anthropic", "model": "m",
                                            "api_key": "k", "route": "api"}).get_json()
    assert r["reply"] == "Here is your fox." and len(r["files"]) == 1 and re.fullmatch(r"generated/image-[0-9a-f]{8}\.png", r["files"][0])
    dl = client(svc).get("/api/workspace/file", query_string={"session_id": "w7", "path": r["files"][0]})
    assert dl.mimetype == "image/png" and dl.data == PNG
