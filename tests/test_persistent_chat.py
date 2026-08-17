from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from persistent_chat import render_prompt


class FakeTokenizer:
    def __init__(self, ids):
        self.ids = ids

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        return self.ids


def test_render_prompt_uses_template_and_preserves_ids():
    assert render_prompt(FakeTokenizer([1, 2, 3]), [{"role": "user", "content": "x"}], 3) == [1, 2, 3]


def test_render_prompt_refuses_truncation():
    with pytest.raises(ValueError, match="Nothing was truncated"):
        render_prompt(FakeTokenizer([1, 2, 3, 4]), [], 3)


def test_render_prompt_rejects_invalid_token():
    with pytest.raises(ValueError, match="outside model vocabulary"):
        render_prompt(FakeTokenizer([152064]), [], 3)
