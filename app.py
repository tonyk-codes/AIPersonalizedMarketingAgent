"""Streamlit front-end for personalized marketing generation."""

from __future__ import annotations

import streamlit as st
from typing import cast

import config
from hf_pipelines import HFProductEncoder, HFProfileEncoder, HFScriptGenerator, HFSloganGenerator, HFVideoGenerator
from interfaces import CustomerProfile, MarketingAssets, ProductInfo
from media_utils import ensure_local_image
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
    border: 1px solid #2f4432;
    border-radius: 14px;
    overflow: hidden;
    background: #0d120d;
}
.preview-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #080a08;
    border-bottom: 1px solid #243625;
    padding: 0.46rem 0.75rem;
    font-size: 0.66rem;
    color: #d6dfd6;
    letter-spacing: 0.06rem;
}
.preview-topbar b {
    color: #ffffff;
}
.preview-hero {
    padding: 0.95rem 0.9rem 0.7rem;
    background: linear-gradient(180deg, #16361e 0%, #6f9972 52%, #b7d4bc 100%);
}
.preview-meta {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    color: #f5f8f5;
}
.preview-date {
    font-size: 0.56rem;
    line-height: 1.15;
    letter-spacing: 0.03rem;
}
.preview-logo {
    font-size: 0.8rem;
    font-style: italic;
    font-weight: 900;
    letter-spacing: 0.04rem;
    transform: skew(-8deg);
}
.preview-headline {
    margin-top: 0.4rem;
    font-size: 1.5rem;
    line-height: 0.98;
    color: #ffffff;
    letter-spacing: 0.02rem;
    font-weight: 800;
    text-shadow: 0 3px 20px rgba(9, 16, 10, 0.45);
}
.preview-subline {
    margin-top: 0.38rem;
    color: #f4f8f4;
    font-size: 0.64rem;
    letter-spacing: 0.01rem;
    opacity: 0.95;
}
.preview-product {
    border-top: 1px solid rgba(255, 255, 255, 0.14);
    background: radial-gradient(circle at 14% 10%, #172318, #090f09 72%);
    padding: 0.62rem;
}
.preview-product-caption {
    display: flex;
    justify-content: space-between;
    gap: 0.4rem;
    color: #f3f8f3;
    margin-top: 0.4rem;
    font-size: 0.65rem;
}
.preview-video {
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    background: #030603;
    padding: 0.5rem;
}
.preview-empty {
    border: 1px dashed #4d6b52;
    border-radius: 10px;
    min-height: 175px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #c5d8c8;
    font-size: 0.8rem;
    text-align: center;
    padding: 0.8rem;
}
.preview-bottom {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.45rem;
    flex-wrap: wrap;
    padding: 0.65rem;
    background: #090d09;
}
.preview-chip {
    border: 1px solid #3b5340;
    border-radius: 7px;
    color: #dceddc;
    padding: 0.32rem 0.44rem;
    font-size: 0.66rem;
    text-align: center;
    background: linear-gradient(180deg, #172219, #111911);
}
.nike-page {
    background: #f5f5f5;
    color: #111111;
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #e4e4e4;
}
.nike-store-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.85rem 1rem;
    background: #ffffff;
    border-bottom: 1px solid #e7e7e7;
}
.nike-store-brand {
    font-weight: 900;
    letter-spacing: 0.22rem;
    font-size: 1rem;
}
.nike-store-menu {
    display: flex;
    gap: 0.9rem;
    font-size: 0.74rem;
    color: #242424;
}
.nike-store-hero-copy {
    padding: 1.05rem 1rem 0.9rem;
    background: linear-gradient(135deg, #101010 0%, #2f2f2f 38%, #5b5b5b 100%);
    color: #ffffff;
}
.nike-store-kicker {
    font-size: 0.7rem;
    letter-spacing: 0.12rem;
    text-transform: uppercase;
    opacity: 0.85;
}
.nike-store-title {
    margin-top: 0.45rem;
    font-size: 1.9rem;
    line-height: 0.98;
    font-weight: 800;
}
.nike-store-description {
    margin-top: 0.45rem;
    font-size: 0.8rem;
    line-height: 1.5;
    max-width: 26rem;
    color: #f1f1f1;
}
.nike-hero-placeholder {
    margin: 0 1rem 1rem;
    border-radius: 14px;
    min-height: 200px;
    background: linear-gradient(135deg, #1a1a1a, #474747);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1.2rem;
    text-align: center;
    color: #ffffff;
}
.nike-pipeline-info {
    margin: 0 1rem 1rem;
    padding: 0.85rem 0.95rem;
    border-radius: 12px;
    background: #ffffff;
    color: #444444;
    font-size: 0.78rem;
    line-height: 1.45;
    border: 1px solid #ececec;
}
.nike-pipeline-info b {
    color: #111111;
}
.nike-products-wrap {
    padding: 0 1rem 1rem;
}
.nike-products-title {
    color: #111111;
    font-weight: 700;
    font-size: 0.88rem;
    margin-bottom: 0.75rem;
}
.nike-product-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.75rem;
}
.nike-product-card {
    background: #ffffff;
    border: 1px solid #ececec;
    border-radius: 12px;
    overflow: hidden;
}
.nike-product-card img {
    width: 100%;
    height: 132px;
    object-fit: cover;
    display: block;
    background: #f1f1f1;
}
.nike-product-body {
    padding: 0.7rem;
}
.nike-product-name {
    font-size: 0.74rem;
    font-weight: 700;
    color: #111111;
    line-height: 1.35;
}
.nike-product-meta {
    margin-top: 0.25rem;
    font-size: 0.68rem;
    color: #757575;
}
.nike-product-price {
    margin-top: 0.32rem;
    font-size: 0.72rem;
    font-weight: 700;
    color: #111111;
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


def build_price_placeholder(product: ProductInfo) -> str:
    base_prices = {
        "Running": 899,
        "Lifestyle": 799,
        "Basketball": 999,
        "Training": 879,
    }
    amount = base_prices.get(product.category, 859)
    return f"HK${amount}"


def render_nike_preview_page(
    video_path: str | None,
    assets: MarketingAssets | None,
    profile: CustomerProfile | None,
    product: ProductInfo | None,
) -> None:
    slogan = assets.slogan if assets else "Move in your own way"
    headline = assets.headline if assets else "Profile-aware creative, generated through multi-stage machine learning pipelines."
    customer_name = profile.name if profile else "your customer"
    selected_label = product.name if product else "selected Nike shoe"

    st.markdown(
        f"""
<div class="nike-page">
    <div class="nike-store-hero-copy">
        <div class="nike-store-title">{slogan}</div>
        <div class="nike-store-description">{headline} Built for {customer_name} and centered on {selected_label}.</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if video_path:
        st.video(video_path)
    else:
        st.markdown(
            """
<div class="nike-hero-placeholder">
    <div>
        <h3>Your personalized Nike shoe ad will appear here.</h3>
        <p>Fill in your profile on the left, choose a shoe, and generate a cinematic 480p front-page banner.</p>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
<div class="nike-page" style="margin-top:-0.2rem; border-top-left-radius:0; border-top-right-radius:0;">
    <div class="nike-pipeline-info">
        <b>Pipeline:</b> profile embedding -&gt; product embedding -&gt; personalized slogan (Flan-T5) -&gt; personalized script (Flan-T5) -&gt; cinematic 480p video (Wan TI2V) -&gt; final frame with slogan and customer name.
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
    render_nike_preview_page(
        video_path=cast(str | None, st.session_state.get("video_path")),
        assets=cast(MarketingAssets | None, st.session_state.get("assets")),
        profile=cast(CustomerProfile | None, st.session_state.get("profile")),
        product=cast(ProductInfo | None, st.session_state.get("product")),
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
