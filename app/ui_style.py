"""Presentation layer for the Streamlit app.

Kept out of `streamlit_app.py` so the application logic stays readable: that
file should be about retrieval, answers and state, not about pixels.

Two rules the styling follows:

  * Colour carries meaning, never decoration. Teal marks a verified citation,
    amber a refusal, red a fabricated reference. Someone scanning an answer
    should be able to tell those apart without reading a word.
  * Selectors use `data-testid` hooks only. Streamlit's generated class names
    change between releases; the test ids are the documented-stable surface,
    so this survives an upgrade instead of silently unstyling itself.
"""

from __future__ import annotations

import streamlit as st

CSS = """
<style>
  :root{
    --ink:#1b1a17; --muted:#6d6a60; --line:#e2ded4;
    --panel:#fcfcfa; --sunk:#f1efe9;
    --accent:#1a6a5a; --accent-soft:#e6f0ed;
    --warn:#a8621a; --warn-soft:#fbf1e4;
    --bad:#a33a2a; --bad-soft:#fbecea;
  }
  /* No prefers-color-scheme block here, deliberately.
     Streamlit's theme is pinned to light in .streamlit/config.toml, and it
     does NOT follow the OS. A media query here followed the OS instead, so on
     a machine set to dark mode these tokens went dark while Streamlit kept
     painting dark text -- producing an unreadable dark-on-dark chat bubble.
     One source of truth: the config file. To offer dark mode, remove `base`
     from config.toml so Streamlit follows the OS, and reinstate the query. */

  /* ---- rhythm ------------------------------------------------------ */
  .block-container{ max-width: 46rem; padding-top: 2.2rem; }
  h1, h2, h3{ letter-spacing:-.015em; font-weight:640; }
  p, li{ line-height:1.68; }

  /* ---- chat --------------------------------------------------------- */
  [data-testid="stChatMessage"]{
    background: transparent;
    border-bottom: 1px solid var(--line);
    border-radius: 0;
    padding: 1.1rem .2rem 1.3rem;
    gap: .85rem;
  }
  [data-testid="stChatMessage"]:last-of-type{ border-bottom: none; }
  /* the user's own turn, set back so the answer leads */
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
    background: var(--sunk);
    border: 1px solid var(--line);
    border-radius: .7rem;
    padding: .8rem 1rem;
    margin-bottom: .4rem;
  }
  [data-testid="stChatMessageAvatarUser"],
  [data-testid="stChatMessageAvatarAssistant"]{ box-shadow:none; }

  /* ---- citations ---------------------------------------------------- */
  /* A verified citation reads as a source you can follow. */
  [data-testid="stChatMessage"] a{
    color: var(--accent);
    background: var(--accent-soft);
    border: 1px solid transparent;
    border-radius: 1rem;
    padding: .05rem .5rem;
    font-size: .86em;
    font-weight: 560;
    text-decoration: none;
    white-space: nowrap;
  }
  [data-testid="stChatMessage"] a:hover{
    border-color: var(--accent);
  }
  /* A citation the validator could NOT match stays as plain code -- never
     dressed up as a link, because it points at nothing. */
  [data-testid="stChatMessage"] code{
    background: var(--bad-soft);
    color: var(--bad);
    border: 1px dashed var(--bad);
    border-radius: .35rem;
    padding: .05rem .35rem;
    font-size: .84em;
  }

  /* ---- retrieved-passage inspector ---------------------------------- */
  [data-testid="stExpander"]{
    border: 1px solid var(--line);
    border-radius: .7rem;
    background: var(--panel);
  }
  [data-testid="stExpander"] summary{
    font-size: .87rem;
    color: var(--muted);
    font-weight: 560;
  }
  [data-testid="stExpander"] [data-testid="stText"]{
    font-size: .8rem;
    line-height: 1.55;
    color: var(--muted);
  }

  /* ---- sidebar ------------------------------------------------------ */
  [data-testid="stSidebar"]{ border-right: 1px solid var(--line); }
  [data-testid="stSidebar"] .stButton button{
    text-align: left;
    justify-content: flex-start;
    font-weight: 460;
    border-color: transparent;
    background: transparent;
    padding: .3rem .55rem;
    min-height: 0;
  }
  [data-testid="stSidebar"] .stButton button:hover{
    background: var(--accent-soft);
    color: var(--accent);
    border-color: transparent;
  }
  [data-testid="stSidebar"] [data-testid="stProgress"] > div > div > div{
    background: var(--accent);
  }

  /* ---- controls ----------------------------------------------------- */
  .stButton button{ border-radius: .55rem; font-weight: 520; }
  [data-testid="stChatInput"]{
    border-radius: .8rem;
    border-color: var(--line);
  }
  [data-testid="stChatInput"]:focus-within{ border-color: var(--accent); }

  /* ---- callouts ----------------------------------------------------- */
  [data-testid="stAlert"]{ border-radius: .6rem; border-width: 0 0 0 3px; }

  /* ---- example prompts on the empty state --------------------------- */
  .example-grid .stButton button{
    width: 100%;
    text-align: left;
    justify-content: flex-start;
    border: 1px solid var(--line);
    background: var(--panel);
    color: var(--muted);
    font-weight: 440;
    padding: .6rem .75rem;
    line-height: 1.4;
    white-space: normal;
    height: 100%;
  }
  .example-grid .stButton button:hover{
    border-color: var(--accent);
    color: var(--accent);
    background: var(--accent-soft);
  }

  /* ---- misc --------------------------------------------------------- */
  [data-testid="stStatusWidget"]{ display:none; }
  footer, #MainMenu{ visibility:hidden; }
  .muted{ color: var(--muted); font-size: .87rem; }
  .pill{
    display:inline-block; padding:.1rem .5rem; border-radius:1rem;
    font-size:.75rem; font-weight:560; border:1px solid var(--line);
    color:var(--muted); background:var(--sunk); margin-right:.3rem;
  }
  .pill.ok{ color:var(--accent); background:var(--accent-soft); border-color:transparent; }
  .pill.warn{ color:var(--warn); background:var(--warn-soft); border-color:transparent; }
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def pill(text: str, kind: str = "") -> str:
    return f'<span class="pill {kind}">{text}</span>'
