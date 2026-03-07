"""Streamlit front-end for the AI Script-to-Screen pipeline.

Run with:
    streamlit run app.py
"""

import importlib
import html
import os

import streamlit as st
from dotenv import load_dotenv

from interfaces import ScriptRefiner, StoryboardGenerator, VideoGenerator

# Load .env so API keys are available before any model class is instantiated.
load_dotenv()

# ---------------------------------------------------------------------------
# Model registry – add new backends here as you integrate real AI services.
# ---------------------------------------------------------------------------

_SCRIPT_REFINERS: dict[str, tuple[str, str]] = {
    "mock": ("mock_implementations", "MockScriptRefiner"),
}

_STORYBOARD_GENERATORS: dict[str, tuple[str, str]] = {
    "mock": ("mock_implementations", "MockStoryboardGen"),
}

_VIDEO_GENERATORS: dict[str, tuple[str, str]] = {
    "mock": ("mock_implementations", "MockVideoGenerator"),
}


def _load_class(registry: dict, model_name: str):
    """Import and return the class registered under *model_name*."""
    if model_name not in registry:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(registry.keys())}"
        )
    module_name, class_name = registry[model_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _inject_styles() -> None:
    """Apply a more polished visual treatment to the Streamlit app."""
    st.markdown(
        """
        <style>
            :root {
                --bg: #f4f1ea;
                --surface: rgba(255, 252, 246, 0.92);
                --surface-strong: #fffdf9;
                --border: rgba(53, 45, 38, 0.12);
                --text: #211c16;
                --muted: #6a625a;
                --accent: #8b5e34;
                --accent-soft: rgba(139, 94, 52, 0.12);
            }

            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(181, 151, 112, 0.18), transparent 26%),
                    linear-gradient(180deg, #f8f4ee 0%, #efe7db 100%);
                color: var(--text);
            }

            [data-testid="stAppViewContainer"] .main .block-container {
                max-width: 1180px;
                padding-top: 2.4rem;
                padding-bottom: 2.5rem;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #221c18 0%, #2f2722 100%);
                border-right: 1px solid rgba(255, 255, 255, 0.08);
            }

            [data-testid="stSidebar"] * {
                color: #f7f1e8;
            }

            .hero-shell,
            .panel,
            .scene-card,
            .stat-card {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 22px;
                box-shadow: 0 18px 45px rgba(55, 43, 31, 0.08);
                backdrop-filter: blur(10px);
            }

            .hero-shell {
                padding: 2rem 2.1rem;
                margin-bottom: 1.3rem;
            }

            .hero-kicker {
                display: inline-block;
                font-size: 0.72rem;
                letter-spacing: 0.16em;
                text-transform: uppercase;
                font-weight: 700;
                color: var(--accent);
                background: var(--accent-soft);
                border-radius: 999px;
                padding: 0.4rem 0.7rem;
                margin-bottom: 1rem;
            }

            .hero-title {
                font-size: clamp(2.2rem, 4vw, 3.5rem);
                line-height: 1.02;
                font-weight: 700;
                margin: 0;
                color: var(--text);
                letter-spacing: -0.04em;
            }

            .hero-copy,
            .section-copy,
            .panel-copy,
            .muted-text {
                color: var(--muted);
                font-size: 1rem;
                line-height: 1.65;
            }

            .panel {
                padding: 1.35rem 1.4rem;
                margin-bottom: 1rem;
            }

            .panel-title,
            .section-title {
                font-size: 1.08rem;
                font-weight: 650;
                margin: 0 0 0.45rem 0;
                color: var(--text);
                letter-spacing: -0.02em;
            }

            .stat-card {
                padding: 1rem 1.1rem;
                height: 100%;
            }

            .stat-label {
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: var(--muted);
                margin-bottom: 0.45rem;
            }

            .stat-value {
                font-size: 1.55rem;
                font-weight: 700;
                color: var(--text);
                letter-spacing: -0.03em;
            }

            .scene-card {
                padding: 1.1rem 1.2rem;
                margin-bottom: 0.9rem;
            }

            .scene-label {
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.14em;
                color: var(--accent);
                margin-bottom: 0.45rem;
                font-weight: 700;
            }

            .stButton > button {
                border-radius: 999px;
                border: 1px solid transparent;
                background: linear-gradient(135deg, #8b5e34 0%, #67472c 100%);
                color: #fff9f2;
                font-weight: 650;
                min-height: 3rem;
            }

            .stButton > button:hover {
                border-color: rgba(255, 255, 255, 0.12);
                background: linear-gradient(135deg, #7d542f 0%, #563a24 100%);
            }

            .stTextArea textarea,
            .stSelectbox [data-baseweb="select"] > div {
                border-radius: 16px;
            }

            .stTextArea textarea {
                background: rgba(255, 255, 255, 0.76);
                border: 1px solid rgba(53, 45, 38, 0.12);
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 0.5rem;
            }

            .stTabs [data-baseweb="tab"] {
                background: rgba(255, 255, 255, 0.5);
                border-radius: 999px;
                padding-left: 1rem;
                padding-right: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Script to Screen",
    layout="wide",
)

_inject_styles()

# ---------------------------------------------------------------------------
# Sidebar – pipeline configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Configuration")
    st.divider()

    st.subheader("Pipeline Models")
    script_model = st.selectbox(
        "Script Refiner",
        list(_SCRIPT_REFINERS.keys()),
        help="Model used to split raw text into structured scene descriptions.",
    )
    storyboard_model = st.selectbox(
        "Storyboard Generator",
        list(_STORYBOARD_GENERATORS.keys()),
        help="Model used to generate a visual reference image for each scene.",
    )
    video_model = st.selectbox(
        "Video Generator",
        list(_VIDEO_GENERATORS.keys()),
        help="Model used to assemble the storyboard images into a video.",
    )

    st.divider()
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Backend Registry</div>
            <div class="panel-copy">
                Add new AI backends by registering them in the model registry dictionaries in app.py.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.markdown(
    """
    <section class="hero-shell">
        <div class="hero-kicker">Creative Pipeline</div>
        <h1 class="hero-title">AI Script to Screen</h1>
        <p class="hero-copy">
            Turn raw narrative text into structured scenes, storyboard frames, and a final video output
            from a single production workspace.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

intro_col, workflow_col = st.columns([1.35, 1], gap="large")

with intro_col:
    st.markdown(
        """
        <div class="panel">
            <div class="section-title">Project Brief</div>
            <div class="section-copy">
                Paste a script, treatment, or short story and run the pipeline to break it into scenes,
                generate visual references, and produce a draft video artifact.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with workflow_col:
    stage_a, stage_b, stage_c = st.columns(3, gap="small")
    with stage_a:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-label">Stage 1</div>
                <div class="stat-value">Refine</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stage_b:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-label">Stage 2</div>
                <div class="stat-value">Storyboard</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stage_c:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-label">Stage 3</div>
                <div class="stat-value">Video</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

raw_text = st.text_area(
    "Enter your script or story",
    placeholder=(
        "A robot explores a jungle. "
        "It discovers an ancient ruin. "
        "The robot makes contact with alien life."
    ),
    height=200,
)

run_btn = st.button("Run Pipeline", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

if run_btn:
    if not raw_text.strip():
        st.warning("Please enter some text before running the pipeline.")
        st.stop()

    # Instantiate the selected backends.
    assert script_model is not None
    assert storyboard_model is not None
    assert video_model is not None
    refiner: ScriptRefiner = _load_class(_SCRIPT_REFINERS, script_model)()
    storyboard_gen: StoryboardGenerator = _load_class(_STORYBOARD_GENERATORS, storyboard_model)()
    video_gen: VideoGenerator = _load_class(_VIDEO_GENERATORS, video_model)()

    # ------------------------------------------------------------------
    # Step 1 – Script refinement
    # ------------------------------------------------------------------
    with st.status("Step 1: Refining script", expanded=True) as step1:
        scenes = refiner.refine(raw_text)
        step1.update(
            label=f"Step 1 complete: script refined into {len(scenes)} scene(s)",
            state="complete",
        )

    summary_cols = st.columns(3, gap="small")
    with summary_cols[0]:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Scenes</div>
                <div class="stat-value">{len(scenes)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for i, scene in enumerate(scenes, 1):
        scene_text = html.escape(scene).replace("\n", "<br>")
        st.markdown(
            f"""
            <div class="scene-card">
                <div class="scene-label">Scene {i:02d}</div>
                <div>{scene_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ------------------------------------------------------------------
    # Step 2 – Storyboard generation
    # ------------------------------------------------------------------
    with st.status("Step 2: Generating storyboard", expanded=True) as step2:
        image_paths = storyboard_gen.generate(scenes)
        step2.update(
            label=f"Step 2 complete: {len(image_paths)} storyboard image(s) generated",
            state="complete",
        )

    with summary_cols[1]:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Storyboard Frames</div>
                <div class="stat-value">{len(image_paths)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if image_paths:
        num_cols = min(len(image_paths), 4)
        cols = st.columns(num_cols)
        for idx, path in enumerate(image_paths):
            with cols[idx % num_cols]:
                st.caption(f"Scene {idx + 1}")
                if os.path.exists(path):
                    st.image(path, use_column_width="auto")
                else:
                    st.info(f"`{path}`\n\n*(placeholder – no file on disk)*")
    else:
        st.info("No storyboard images were produced.")

    st.divider()

    # ------------------------------------------------------------------
    # Step 3 – Video generation
    # ------------------------------------------------------------------
    with st.status("Step 3: Generating video", expanded=True) as step3:
        video_path = video_gen.generate(image_paths)
        step3.update(
            label=f"Step 3 complete: video generated at {video_path}",
            state="complete",
        )

    with summary_cols[2]:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Output</div>
                <div class="stat-value">Ready</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Output Video")
    if os.path.exists(video_path):
        st.video(video_path)
    else:
        st.success(
            f"Pipeline complete! Output: `{video_path}`\n\n"
            "*(Replace the mock backends with real AI services to produce an actual video file.)*"
        )
