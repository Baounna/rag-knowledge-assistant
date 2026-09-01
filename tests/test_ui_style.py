"""The stylesheet must agree with the pinned Streamlit theme."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ui_style import CSS, pill  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_stylesheet_does_not_follow_the_os_when_the_theme_is_pinned():
    """Regression: the CSS had a `prefers-color-scheme: dark` block while
    config.toml pinned `base = "light"`. Streamlit does not follow the OS, so
    on a dark-mode machine the tokens flipped dark, Streamlit kept painting
    dark text, and the user's chat bubble rendered dark-on-dark -- invisible.

    Either both follow the OS or neither does. This asserts they agree.
    """
    config = (ROOT / ".streamlit/config.toml").read_text()
    pinned = re.search(r'^base\s*=\s*"(light|dark)"', config, re.M)
    # Strip comments first: the explanation of WHY there is no media query
    # naturally mentions it, and matching that would fail the very fix it
    # documents.
    active = re.sub(r"/\*.*?\*/", "", CSS, flags=re.S)
    if pinned:
        assert "prefers-color-scheme" not in active, (
            f'config.toml pins base="{pinned.group(1)}" but the stylesheet '
            f"follows the OS colour scheme; they will disagree and produce "
            f"unreadable text"
        )


def test_every_colour_token_is_defined():
    """A var() with no definition silently renders as nothing."""
    defined = set(re.findall(r"(--[a-z-]+)\s*:", CSS))
    used = set(re.findall(r"var\((--[a-z-]+)\)", CSS))
    assert not (used - defined), f"undefined tokens: {sorted(used - defined)}"


def test_user_and_assistant_turns_are_distinguishable():
    assert "stChatMessageAvatarUser" in CSS


def test_pill_helper_marks_its_kind():
    assert 'class="pill ok"' in pill("x", "ok")
    assert 'class="pill "' in pill("x")
