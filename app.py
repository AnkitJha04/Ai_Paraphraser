"""Streamlit interface for Rephrase AI."""

from __future__ import annotations

import streamlit as st

from modules.paraphraser import (
    DEFAULT_MODEL,
    get_installed_models,
    lexical_overlap,
    stream_paraphrase_text,
    text_statistics,
)

st.set_page_config(page_title="Rephrase AI", page_icon="✦", layout="wide")

st.markdown(
    """
    <style>
    .block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem; }
    [data-testid="stMetric"] { border-left: 2px solid #4f46e5; padding-left: .75rem; }
    div[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.18); }
    .app-kicker { color: #6366f1; font-size: .78rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; }
    .app-subtitle { color: rgba(128,128,128,.95); font-size: 1rem; margin-bottom: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

TONES = ["Neutral", "Professional", "Friendly", "Formal", "Academic", "Conversational", "Persuasive", "Confident"]
TASKS = ["Paraphrase", "Simplify", "Make concise", "Expand with context", "Improve clarity", "Fix grammar"]
LENGTHS = ["Keep similar length", "Shorter", "A little longer", "No length preference"]


def initialise_state() -> None:
    defaults = {
        "editor": "",
        "results": [],
        "history": [],
        "last_original": "",
        "last_settings": {},
        "result_revision": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_workspace() -> None:
    """Widget callback: updates state before Streamlit creates editor widgets."""
    st.session_state.editor = ""
    st.session_state.results = []
    st.session_state.last_original = ""
    st.session_state.last_settings = {}
    st.session_state.result_revision += 1


def load_history(item: dict) -> None:
    """Widget callback used to safely restore an earlier generation."""
    st.session_state.editor = item["original"]
    st.session_state.results = item["results"]
    st.session_state.last_original = item["original"]
    st.session_state.last_settings = item["settings"]
    st.session_state.result_revision += 1


initialise_state()

try:
    installed_models = get_installed_models()
    connection_error = ""
except Exception as exc:  # Connection errors differ between Ollama versions/OSes.
    installed_models = []
    connection_error = str(exc)

with st.sidebar:
    st.title("✦ Rephrase AI")
    st.caption("A local writing workspace powered by Ollama")
    st.divider()
    st.subheader("Model")
    if connection_error:
        st.error("Ollama is unavailable")
        with st.expander("Connection details"):
            st.code(connection_error, language=None)
        st.caption("Start Ollama, then reload this page. On Windows, opening the Ollama app is normally enough.")
        model = st.text_input("Model tag", value=DEFAULT_MODEL, disabled=True)
    elif installed_models:
        default_index = installed_models.index(DEFAULT_MODEL) if DEFAULT_MODEL in installed_models else 0
        model = st.selectbox("Installed model", installed_models, index=default_index)
        st.success(f"Connected · {len(installed_models)} model{'s' if len(installed_models) != 1 else ''} available")
    else:
        st.warning("Ollama is connected, but no models are installed.")
        st.code("ollama pull llama3.1:latest", language="bash")
        model = st.text_input("Model tag", value=DEFAULT_MODEL)

    st.divider()
    st.subheader("Generation")
    temperature = st.slider("Creativity", 0.0, 1.0, 0.25, 0.05, help="Lower values give more conservative rewrites.")
    context_size = st.select_slider("Context window", options=[2048, 4096, 8192], value=4096)
    st.caption("Text stays on this machine and is sent only to your local Ollama service.")

st.markdown('<div class="app-kicker">Local writing assistant</div>', unsafe_allow_html=True)
st.title("Rewrite with control, not guesswork.")
st.markdown('<div class="app-subtitle">Choose the outcome and constraints, then review a live local-model draft before exporting it.</div>', unsafe_allow_html=True)

editor_column, controls_column = st.columns([1.65, 1], gap="large")
with editor_column:
    st.subheader("Source text")
    original = st.text_area(
        "Source text",
        key="editor",
        height=280,
        placeholder="Paste a sentence, email, paragraph, or report section here…",
        label_visibility="collapsed",
    )
    statistics = text_statistics(original)
    stat_columns = st.columns(3)
    stat_columns[0].metric("Words", statistics["words"])
    stat_columns[1].metric("Characters", statistics["characters"])
    stat_columns[2].metric("Sentences", statistics["sentences"])

with controls_column:
    st.subheader("Rewrite settings")
    task = st.selectbox("What do you need?", TASKS)
    tone = st.selectbox("Tone", TONES)
    length = st.selectbox("Length", LENGTHS)
    maximum_words = st.number_input(
        "Maximum output words (optional)", min_value=0, max_value=5000, value=0, step=10,
        help="Enter 0 for no hard limit. Any other value is enforced after generation.",
    )
    alternatives = st.radio("Drafts", options=[1, 2, 3], index=0, horizontal=True, help="More drafts take proportionally longer.")
    audience = st.selectbox("Audience", ["General readers", "Students", "Teachers and academics", "Workplace", "Customers"])
    preserve_details = st.checkbox("Lock names, numbers, and facts", value=True)

with st.expander("Strict requirements and protected terms"):
    guidance_col, terms_col = st.columns(2)
    with guidance_col:
        custom_instruction = st.text_area(
            "Strict requirements", height=100,
            placeholder="Example: Use natural language. Give exactly one sentence, close to 25 words.",
            help="These requirements override the default tone and length guidance. The app checks common sentence and word-count requests.",
        )
    with terms_col:
        terms_to_keep = st.text_area("Terms that must remain unchanged", placeholder="One or more names, terms, or phrases", height=100)

action_col, clear_col, note_col = st.columns([1.1, .65, 1.75], vertical_alignment="center")
with action_col:
    generate = st.button(
        "Generate rewrite",
        type="primary",
        use_container_width=True,
        disabled=bool(connection_error),
    )
with clear_col:
    st.button("Clear", use_container_width=True, on_click=clear_workspace)
with note_col:
    if not connection_error and not installed_models:
        st.caption("Install a model before generating a rewrite.")
    else:
        st.caption("Generation streams from Ollama as it is produced.")

if generate:
    if not original.strip():
        st.warning("Add some source text before generating a rewrite.")
    elif len(original) > 16_000:
        st.warning("This workspace accepts up to 16,000 characters at a time. Please shorten the source text.")
    elif not installed_models:
        st.warning("Install an Ollama model first, then reload the page.")
    else:
        results: list[str] = []
        progress = st.progress(0, text="Preparing local model…")
        live_area = st.empty()
        try:
            for index in range(int(alternatives)):
                progress.progress(index / int(alternatives), text=f"Writing draft {index + 1} of {alternatives}…")
                with live_area.container():
                    st.caption(f"Live draft {index + 1}")
                    preview = st.empty()

                def update_preview(value: str, target=preview) -> None:
                    target.text(value + "▍")

                result = stream_paraphrase_text(
                    text=original, task=task, tone=tone, length=length, audience=audience,
                    model=model, temperature=temperature, num_ctx=context_size,
                    preserve_details=preserve_details, custom_instruction=custom_instruction,
                    terms_to_keep=terms_to_keep, variation_index=index, maximum_words=maximum_words,
                    on_chunk=update_preview,
                )
                results.append(result)
                live_area.empty()
                progress.progress((index + 1) / int(alternatives), text=f"Draft {index + 1} complete.")

            st.session_state.results = results
            st.session_state.last_original = original
            st.session_state.result_revision += 1
            st.session_state.last_settings = {
                "task": task, "tone": tone, "length": length, "maximum words": maximum_words or "No limit",
                "audience": audience, "model": model,
            }
            st.session_state.history.insert(0, {
                "original": original, "results": results, "settings": st.session_state.last_settings.copy(),
            })
            st.session_state.history = st.session_state.history[:8]
            progress.empty()
            st.success("Rewrite ready.")
        except Exception as exc:
            live_area.empty()
            progress.empty()
            st.error("The rewrite could not be generated.")
            st.caption("Check that the selected model is still installed and Ollama is running.")
            with st.expander("Technical details"):
                st.code(str(exc), language=None)

if st.session_state.results:
    st.divider()
    st.subheader("Your drafts")
    settings = st.session_state.last_settings
    st.caption(" · ".join(f"{name.title()}: {value}" for name, value in settings.items()))
    source = st.session_state.last_original
    revision = st.session_state.result_revision
    tabs = st.tabs([f"Draft {index + 1}" for index in range(len(st.session_state.results))])
    for index, (tab, result) in enumerate(zip(tabs, st.session_state.results), start=1):
        with tab:
            source_col, result_col = st.columns(2, gap="large")
            with source_col:
                st.caption("SOURCE")
                st.text_area("Source", value=source, height=240, disabled=True, key=f"source_{revision}_{index}", label_visibility="collapsed")
            with result_col:
                st.caption("REWRITE")
                st.text_area("Rewrite", value=result, height=240, disabled=True, key=f"result_{revision}_{index}", label_visibility="collapsed")
            metric_col, download_col = st.columns([1, 1])
            with metric_col:
                overlap = lexical_overlap(source, result)
                st.caption(f"Word overlap: {overlap:.0f}% — a wording indicator, not a plagiarism or meaning score.")
            with download_col:
                st.download_button("Download this draft", result, f"rephrase-draft-{index}.txt", "text/plain", key=f"download_{index}")

    all_drafts = "\n\n".join(f"DRAFT {index}\n{result}" for index, result in enumerate(st.session_state.results, start=1))
    st.download_button("Download all drafts", f"SOURCE\n{source}\n\n{all_drafts}", "rephrase-ai-drafts.txt", "text/plain")

with st.expander(f"Session history ({len(st.session_state.history)})"):
    if not st.session_state.history:
        st.caption("Generated drafts are kept here until you close this browser session.")
    for index, item in enumerate(st.session_state.history):
        heading = f"{index + 1}. {item['settings']['task']} · {item['settings']['tone']}"
        st.markdown(f"**{heading}**")
        st.caption(item["original"][:170] + ("…" if len(item["original"]) > 170 else ""))
        st.button("Load", key=f"history_{index}", on_click=load_history, args=(item,))
        if index < len(st.session_state.history) - 1:
            st.divider()

st.caption("Rephrase AI runs locally with Streamlit and Ollama. Review generated writing before using it in important work.")
