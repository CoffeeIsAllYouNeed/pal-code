import pytest

from pal.redact import redact


@pytest.mark.parametrize("secret", [
    "sk-abcdefghijklmnop", "sk-ant-api03-AbCdEfGhIjKl_mnop", "AIzaSyA1b2C3d4E5f6G7h8", "AQ.FakeKeyForTestsOnly_0123456789abcdef",
    "AKIAIOSFODNN7EXAMPLE", "Bearer abcdefghijklmnopqrstuvwx"])
def test_key_shaped_strings_are_hidden_even_when_the_key_is_not_known(secret):
    out = redact(f"401: bad key {secret} was rejected")
    assert secret not in out and "***" in out and out.startswith("401: bad key") and out.endswith("was rejected")


def test_known_keys_are_hidden_exactly_whatever_their_shape_and_empty_ones_are_ignored():
    assert redact("the key is hunter2-xyz here", "hunter2-xyz", None, "") == "the key is *** here"
    assert redact("nothing secret in this sentence about sky and ask") == "nothing secret in this sentence about sky and ask"


@pytest.mark.parametrize("text,gone", [
    ("GET /search?q=cats&api_key=abc123xyz HTTP/1.1", "abc123xyz"), ("https://x.test/v1?appid=DEMO-1234&input=pi", "DEMO-1234"),
    ("failed: token=ghp_whatever and more", "ghp_whatever"), ("?key=plain-short&x=1", "plain-short"), ("password=hunter2 ok", "hunter2")])
def test_secret_looking_query_pairs_are_hidden_but_the_rest_of_the_url_stays(text, gone):
    out = redact(text)
    assert gone not in out and "=***" in out and ("q=cats" in out or "input=pi" in out or "x=1" in out or "ok" in out or "HTTP/1.1" in out)


def test_ordinary_text_with_equals_signs_is_left_alone():
    assert redact("a=1 and b=2 and monkey=banana") == "a=1 and b=2 and monkey=banana"
