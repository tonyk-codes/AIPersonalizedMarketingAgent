"""Streamlit front-end for personalized marketing generation."""

from __future__ import annotations

import streamlit as st
from typing import cast

import config
from hf_pipelines import HFProductEncoder, HFProfileEncoder, HFScriptGenerator, HFSloganGenerator, HFVideoGenerator
from interfaces import CustomerProfile, MarketingAssets, ProductInfo
from mock_implementations import MockProductEncoder, MockProfileEncoder, MockScriptGenerator, MockSloganGenerator, MockVideoGenerator
from nike_catalog import get_product_by_name, list_products

config.ensure_artifact_dirs()

st.set_page_config(page_title="Machine Learning for Personalized Marketing", layout="wide")

hf_token = st.secrets.get("HF_TOKEN", "")
if hf_token:
    config.HF_TOKEN = hf_token

st.markdown(
    """
<style>
.stApp {
  background: radial-gradient(circle at 20% 20%, #1f1f1f, #0b0b0b 55%);
  color: #f4f4f4;
}
.block-container {
  padding-top: 1.3rem;
  padding-bottom: 2rem;
}
.copy-card {
  margin-top: 0.8rem;
  border: 1px solid #2f2f2f;
  border-radius: 14px;
  padding: 0.9rem;
  background: #121212;
}
.copy-headline {
  color: #ef3b3a;
  font-size: 1.15rem;
  font-weight: 700;
}
.pipeline-list {
    margin: 0.8rem 0 1rem;
    padding: 0.8rem 0.95rem;
    border: 1px solid #2b2b2b;
    border-radius: 14px;
    background: linear-gradient(180deg, #111, #0b0b0b);
}
.pipeline-list ol {
    margin: 0.35rem 0 0;
    padding-left: 1.2rem;
}
.pipeline-list li {
    margin: 0.28rem 0;
    color: #d7d7d7;
}
.preview-empty {
    border: 1px dashed #4d6b52;
    border-radius: 10px;
    width: 100%;
    min-height: 220px;
    aspect-ratio: 16 / 9;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #c5d8c8;
    font-size: 0.8rem;
    text-align: center;
    padding: 0.8rem;
}
.right-preview [data-testid="stVideo"] {
    width: 100% !important;
}
.right-preview [data-testid="stVideo"] > div {
    width: 100% !important;
}
.right-preview [data-testid="stVideo"] video,
.right-preview [data-testid="stVideo"] iframe {
    width: 100% !important;
    height: auto !important;
    display: block;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_catalog() -> list[ProductInfo]:
    return list_products()


def init_state() -> None:
    defaults = {
        "profile": None,
        "product": None,
        "assets": None,
        "video_path": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def get_backends() -> tuple[
    HFProfileEncoder | MockProfileEncoder,
    HFProductEncoder | MockProductEncoder,
    HFSloganGenerator | MockSloganGenerator,
    HFScriptGenerator | MockScriptGenerator,
    HFVideoGenerator | MockVideoGenerator,
    bool,
]:
    use_mock = config.FORCE_MOCK_MODE or not config.HF_TOKEN
    if use_mock:
        return MockProfileEncoder(), MockProductEncoder(), MockSloganGenerator(), MockScriptGenerator(), MockVideoGenerator(), True
    return HFProfileEncoder(), HFProductEncoder(), HFSloganGenerator(), HFScriptGenerator(), HFVideoGenerator(), False


def validate_name(name: str) -> bool:
    return bool(name and name.strip())


def render_nike_preview_page(
    video_path: str | None,
) -> None:
    if video_path:
        st.video(video_path)
    else:
        st.markdown(
            """
<div class="preview-empty">
    <div>
        <h3>Your generated marketing video will appear here.</h3>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )


PIPELINE_STEPS = [
    "Encoding customer profile",
    "Encoding Nike product features",
    "Generating personalized slogan",
    "Generating personalized script",
    "Generating cinematic 480p video",
]


init_state()
products = load_catalog()
product_names = [item.name for item in products]

st.title("Machine Learning for Personalized Marketing")
st.caption("Machine learning-driven personalized product campaigns generated through chained Hugging Face text and video pipelines.")

st.markdown(
        """
<div class="pipeline-list">
    <b>Pipeline Overview</b>
    <ol>
        <li>Profile embedding</li>
        <li>Product embedding</li>
        <li>Personalized slogan generation</li>
        <li>Personalized script generation</li>
        <li>Cinematic 480p video composition with final slogan frame</li>
    </ol>
</div>
""",
        unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Customer Profile")
    name = st.text_input("Name", value="")
    age = int(st.number_input("Age", min_value=10, max_value=90, value=25, step=1))
    gender = cast(str, st.selectbox("Gender", ["Male", "Female", "Non-binary", "Prefer not to say"]))
    nationality = st.text_input("Nationality", help="e.g., Hong Kong, USA, Japan")
    language = cast(
        str,
        st.selectbox(
            "Language",
            ["English", "Traditional Chinese", "Simplified Chinese", "Japanese", "Other"],
        ),
    )
    selected_product_name = cast(str, st.selectbox("Product", options=product_names))

    if config.FORCE_MOCK_MODE or not config.HF_TOKEN:
        st.warning("HF token missing or mock mode forced: app runs in deterministic mock mode.")

    generate_clicked = st.button("Generate Marketing Video", type="primary", use_container_width=True)

if generate_clicked:
    if not validate_name(name):
        st.error("Name is required. Please enter a non-empty Name.")
    else:
        selected_product = get_product_by_name(selected_product_name)
        customer = CustomerProfile(
            name=name.strip(),
            age=age,
            gender=gender,
            nationality=nationality.strip(),
            language=language,
            product_id=selected_product.product_id,
        )

        progress = st.progress(0, text="Starting pipeline...")
        log_box = st.empty()
        logs: list[str] = []

        with st.status("Generating personalized Nike ad assets...", expanded=True) as status:
            profile_encoder, product_encoder, slogan_generator, script_generator, video_generator, use_mock = get_backends()

            logs.append("[1/5] Encoding customer profile")
            log_box.code("\n".join(logs), language="text")
            encoded_profile = profile_encoder.encode(customer)
            progress.progress(20, text="Pipeline 1/5: Encoding customer profile")

            logs.append("[2/5] Encoding Nike product features")
            log_box.code("\n".join(logs), language="text")
            encoded_product = product_encoder.encode(selected_product)
            progress.progress(40, text="Pipeline 2/5: Encoding Nike product features")

            logs.append("[3/5] Generating personalized slogan")
            log_box.code("\n".join(logs), language="text")
            slogan = slogan_generator.generate(customer, encoded_profile, selected_product, encoded_product)
            progress.progress(60, text="Pipeline 3/5: Generating personalized slogan")

            logs.append("[4/5] Generating personalized script")
            log_box.code("\n".join(logs), language="text")
            assets = script_generator.generate(customer, encoded_profile, selected_product, encoded_product, slogan)
            assets.final_slogan_text = f"{assets.slogan}, {customer.name}"
            progress.progress(80, text="Pipeline 4/5: Generating personalized script")

            logs.append("[5/5] Generating cinematic 480p video")
            log_box.code("\n".join(logs), language="text")
            video_path = video_generator.generate(customer, selected_product, assets)
            assets.video_path = video_path

            if assets.debug_metadata is None:
                assets.debug_metadata = {}
            assets.debug_metadata["runtime_mode"] = "mock" if use_mock else "huggingface"

            logs.append("[done] Marketing assets generated successfully")
            log_box.code("\n".join(logs), language="text")
            progress.progress(100, text="Pipeline complete")
            status.update(label="Marketing assets ready", state="complete")

        st.session_state["profile"] = customer
        st.session_state["product"] = selected_product
        st.session_state["assets"] = assets
        st.session_state["video_path"] = assets.video_path

assets: MarketingAssets | None = st.session_state.get("assets")

if assets:
    hero_col, side_col = st.columns([2.6, 1.2], gap="large")

    with hero_col:
        st.markdown('<div class="copy-card">', unsafe_allow_html=True)
        st.markdown(f"<div class='copy-headline'>{assets.slogan}</div>", unsafe_allow_html=True)
        st.markdown(f"**{assets.headline}**")
        st.write(assets.script)
        st.markdown("</div>", unsafe_allow_html=True)

    with side_col:
        st.markdown('<div class="right-preview">', unsafe_allow_html=True)
        render_nike_preview_page(
            video_path=cast(str | None, st.session_state.get("video_path")),
        )
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown('<div class="right-preview">', unsafe_allow_html=True)
    render_nike_preview_page(
        video_path=cast(str | None, st.session_state.get("video_path")),
    )
    st.markdown("</div>", unsafe_allow_html=True)

profile_state: CustomerProfile | None = st.session_state.get("profile")
product_state: ProductInfo | None = st.session_state.get("product")
if profile_state and product_state:
    st.divider()
    st.markdown("### Current Personalization")
    st.write(
        {
            "name": profile_state.name,
            "age": profile_state.age,
            "gender": profile_state.gender,
            "nationality": profile_state.nationality,
            "language": profile_state.language,
            "product": product_state.name,
        }
    )
