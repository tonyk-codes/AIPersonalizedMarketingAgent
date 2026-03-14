"""Streamlit front-end for Smart Nike Shoe Ad Studio."""

from __future__ import annotations

import streamlit as st
from typing import cast

import config
from hf_pipelines import HFCopyGenerator, HFProductEncoder, HFProfileEncoder, HFVideoGenerator
from interfaces import CustomerProfile, MarketingAssets, ProductInfo
from media_utils import ensure_local_image
from mock_implementations import MockCopyGenerator, MockProductEncoder, MockProfileEncoder, MockVideoGenerator
from nike_catalog import get_product_by_name, list_products

config.ensure_artifact_dirs()

st.set_page_config(page_title="Smart Nike Shoe Ad Studio", layout="wide")

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
.nike-nav {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: linear-gradient(90deg, #111111, #1a1a1a);
  border: 1px solid #2c2c2c;
  border-radius: 12px;
  padding: 0.8rem 1.2rem;
  margin-bottom: 0.9rem;
}
.nike-brand {
  letter-spacing: 0.14rem;
  font-weight: 900;
  color: #ffffff;
}
.nike-menu {
  color: #d1d1d1;
  font-size: 0.95rem;
}
.hero-shell {
  border: 1px solid #313131;
  border-radius: 14px;
  padding: 0.7rem;
  background: linear-gradient(180deg, #141414, #090909);
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
.tile-grid {
  margin-top: 1rem;
}
.tile-box {
  border: 1px solid #2a2a2a;
  border-radius: 10px;
  padding: 0.65rem;
  background: #101010;
}
.preview-shell {
    border: 1px solid #2f3f2f;
    border-radius: 14px;
    overflow: hidden;
    background: #0a130a;
}
.preview-top {
    background: linear-gradient(180deg, #16351d 0%, #5e8c67 58%, #a6c9ad 100%);
    padding: 0.9rem;
}
.preview-brand {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #f5f8f5;
    font-size: 0.85rem;
    letter-spacing: 0.05rem;
    margin-bottom: 0.7rem;
}
.preview-promo {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 10px;
    margin-bottom: 0.45rem;
    padding: 0.45rem 0.6rem;
    color: #f3f7f3;
    background: rgba(0, 0, 0, 0.22);
    font-size: 0.85rem;
}
.preview-promo b {
    font-size: 1.1rem;
}
.preview-video {
    border-top: 1px solid rgba(255, 255, 255, 0.18);
    background: #000000;
    padding: 0.55rem;
}
.preview-empty {
    border: 1px dashed #4a6650;
    border-radius: 10px;
    min-height: 240px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #c5d8c8;
    font-size: 0.9rem;
    text-align: center;
    padding: 1rem;
}
.preview-bottom {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    padding: 0.7rem 0.9rem 0.95rem;
    background: #0b110c;
}
.preview-chip {
    border: 1px solid #39533f;
    border-radius: 999px;
    color: #d7ead9;
    padding: 0.25rem 0.55rem;
    font-size: 0.78rem;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_catalog() -> list[ProductInfo]:
    return list_products()


@st.cache_data(show_spinner=False)
def cached_local_image(image_path_or_url: str, name: str) -> str:
    return ensure_local_image(image_path_or_url, name_hint=name)


def init_state() -> None:
    defaults = {
        "profile": None,
        "product": None,
        "assets": None,
        "video_path": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def get_backends() -> tuple[HFProfileEncoder | MockProfileEncoder, HFProductEncoder | MockProductEncoder, HFCopyGenerator | MockCopyGenerator, HFVideoGenerator | MockVideoGenerator, bool]:
    use_mock = config.FORCE_MOCK_MODE or not config.HF_TOKEN
    if use_mock:
        return MockProfileEncoder(), MockProductEncoder(), MockCopyGenerator(), MockVideoGenerator(), True
    return HFProfileEncoder(), HFProductEncoder(), HFCopyGenerator(), HFVideoGenerator(), False


def validate_name(name: str) -> bool:
    return bool(name and name.strip())


def render_nav_bar() -> None:
    st.markdown(
        """
<div class="nike-nav">
  <div class="nike-brand">NIKE</div>
  <div class="nike-menu">Men | Women | Kids | Collections | Sale</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_product_tiles(products: list[ProductInfo]) -> None:
    st.markdown('<div class="tile-grid"></div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, product in enumerate(products[:3]):
        with cols[idx]:
            image = cached_local_image(product.image_path_or_url, product.name)
            st.image(image, width=260)
            st.markdown(f"<div class='tile-box'><b>{product.name}</b><br>{product.category}</div>", unsafe_allow_html=True)


def run_pipeline(profile: CustomerProfile, product: ProductInfo) -> MarketingAssets:
    profile_encoder, product_encoder, copy_generator, video_generator, use_mock = get_backends()

    encoded_profile = profile_encoder.encode(profile)
    encoded_product = product_encoder.encode(product)
    assets = copy_generator.generate(profile, encoded_profile, product, encoded_product)

    video_path = video_generator.generate(profile, product, assets)
    assets.video_path = video_path

    if assets.debug_metadata is None:
        assets.debug_metadata = {}
    assets.debug_metadata["runtime_mode"] = "mock" if use_mock else "huggingface"

    return assets


init_state()
products = load_catalog()
product_names = [item.name for item in products]

st.title("Smart Nike Shoe Ad Studio")
st.caption("Personalized Nike shoe video ads built with a 4-stage Hugging Face pipeline.")

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
    notes = st.text_area("Additional notes (optional)", value="")

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
            additional_notes=notes.strip(),
        )

        with st.status("Generating personalized Nike ad assets...", expanded=True) as status:
            st.write("Pipeline 1/4: Encoding customer profile")
            st.write("Pipeline 2/4: Encoding Nike product features")
            st.write("Pipeline 3/4: Generating personalized slogan and script")
            st.write("Pipeline 4/4: Generating video (or fallback banner video)")
            assets = run_pipeline(customer, selected_product)
            status.update(label="Marketing assets ready", state="complete")

        st.session_state["profile"] = customer
        st.session_state["product"] = selected_product
        st.session_state["assets"] = assets
        st.session_state["video_path"] = assets.video_path

render_nav_bar()

hero_col, side_col = st.columns([2.6, 1.2], gap="large")

with hero_col:
    st.markdown('<div class="hero-shell">', unsafe_allow_html=True)
    st.markdown("### Campaign Studio")
    st.write("Use the profile form to generate personalized ad copy and a video for preview.")
    render_product_tiles(products)
    st.markdown("</div>", unsafe_allow_html=True)

    assets: MarketingAssets | None = st.session_state.get("assets")
    if assets:
        st.markdown('<div class="copy-card">', unsafe_allow_html=True)
        st.markdown(f"<div class='copy-headline'>{assets.slogan}</div>", unsafe_allow_html=True)
        st.markdown(f"**{assets.headline}**")
        st.write(assets.script)
        st.markdown("</div>", unsafe_allow_html=True)

with side_col:
        st.markdown(
                """
<div class="preview-shell">
    <div class="preview-top">
        <div class="preview-brand">
            <span>NIKE HONG KONG</span>
            <span>Online Store</span>
        </div>
        <div class="preview-promo">
            <span>2026.03.10 STARTS</span>
            <b>Season Offer</b>
        </div>
        <div class="preview-promo">
            <span>Members Exclusive</span>
            <b>Up To 65% Off</b>
        </div>
    </div>
    <div class="preview-video">
""",
                unsafe_allow_html=True,
        )

        if st.session_state.get("video_path"):
                st.video(st.session_state["video_path"])
        else:
                st.markdown(
                        '<div class="preview-empty">Generate Marketing Video to replace the hero image with your personalized ad video preview.</div>',
                        unsafe_allow_html=True,
                )

        st.markdown(
                """
    </div>
    <div class="preview-bottom">
        <span class="preview-chip">New In</span>
        <span class="preview-chip">Shoes</span>
        <span class="preview-chip">Apparel</span>
        <span class="preview-chip">Member Days</span>
    </div>
</div>
""",
                unsafe_allow_html=True,
        )

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
