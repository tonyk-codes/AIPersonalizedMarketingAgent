"""Hugging Face-backed pipeline implementations for Smart Nike Shoe Ad Studio."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from typing import Any, cast

import streamlit as st
import torch
from huggingface_hub import InferenceClient
from PIL import Image
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
)

import config
from interfaces import (
    CustomerProfile,
    EncodedProduct,
    EncodedProfile,
    MarketingAssets,
    ProductEncoder,
    ProductInfo,
    ProfileEncoder,
    ScriptGenerator,
    SloganGenerator,
    VideoGenerator,
)
from media_utils import (
    compose_final_ad_video,
    create_fallback_banner_video,
    ensure_local_image,
    save_video_bytes,
    sanitize_filename,
)


def _deterministic_embedding(text: str, size: int = 384) -> list[float]:
    """Return deterministic pseudo-embedding for graceful degradation."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    while len(values) < size:
        for byte in digest:
            values.append((byte / 255.0) * 2.0 - 1.0)
            if len(values) == size:
                break
        digest = hashlib.sha256(digest).digest()
    return values


def build_profile_summary(profile: CustomerProfile) -> str:
    """Pure helper used for profile encoding and prompt construction."""
    notes = profile.additional_notes.strip() or "No additional notes"
    return (
        f"Name: {profile.name}; Age: {profile.age}; Gender: {profile.gender}; "
        f"Nationality: {profile.nationality or 'Not specified'}; Language: {profile.language}; "
        f"Product ID: {profile.product_id}; Notes: {notes}."
    )


def _age_group(age: int) -> str:
    if age < 18:
        return "teenager"
    if age < 30:
        return "person in their 20s"
    if age < 40:
        return "person in their 30s"
    if age < 50:
        return "person in their 40s"
    return "mature adult"


def build_slogan_prompt(
    profile_summary: str,
    product: ProductInfo,
    product_attributes: dict[str, str],
) -> str:
    attrs = ", ".join(f"{k}: {v}" for k, v in product_attributes.items()) or "No attributes provided"
    return (
        "You are a Nike marketing slogan expert. "
        "Create ONE short, punchy slogan with at most 10 words. "
        "Return ONLY the slogan text, no explanation, no quotation marks.\n\n"
        f"Customer profile: {profile_summary}\n"
        f"Shoe: {product.name} - {product.category}\n"
        f"Product attributes: {attrs}\n"
        "Make it tailored to the customer and directly relevant to the shoe."
    )


def build_script_prompt(
    profile_summary: str,
    slogan: str,
    product: ProductInfo,
    product_attributes: dict[str, str],
) -> str:
    attrs = ", ".join(f"{k}: {v}" for k, v in product_attributes.items()) or "No attributes provided"
    return (
        "You are a Nike marketing copywriter. Write personalized ad copy in English only. "
        "Create output as strict JSON with keys: headline, script. "
        "headline should be one line. script should be 3-5 sentences and suitable for 15-40 second narration.\n\n"
        f"Customer profile: {profile_summary}\n"
        f"Approved slogan: {slogan}\n"
        f"Product name: {product.name}\n"
        f"Product category: {product.category}\n"
        f"Product audience: {product.gender}\n"
        f"Product attributes: {attrs}\n"
        f"Product URL: {product.product_url}\n\n"
        "Keep the tone aspirational, energetic, and premium. "
        "Mention practical fit for the customer profile. "
        "Future multilingual adaptation note: only output English for now."
    )


def parse_generated_slogan(raw_output: str) -> str:
    raw_output = (raw_output or "").strip()
    if not raw_output:
        return "Move Beyond Limits"

    slogan = raw_output.splitlines()[0].strip().strip('"')
    words = slogan.split()
    if len(words) > 10:
        slogan = " ".join(words[:10])
    return slogan[:80] or "Move Beyond Limits"


def parse_generated_script(raw_output: str) -> tuple[str, str]:
    raw_output = (raw_output or "").strip()
    if not raw_output:
        return (
            "Designed for your next step.",
            "These Nike shoes are tailored to your lifestyle and goals. Move with confidence and style every day.",
        )

    try:
        parsed = json.loads(raw_output)
        headline = str(parsed.get("headline", "Designed for your next step.")).strip()
        script = str(parsed.get("script", "Feel the energy in every stride.")).strip()
        return headline, script
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{.*\}", raw_output, flags=re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            headline = str(parsed.get("headline", "Designed for your next step.")).strip()
            script = str(parsed.get("script", "Feel the energy in every stride.")).strip()
            return headline, script
        except json.JSONDecodeError:
            pass

    lines = [line.strip("-• ") for line in raw_output.splitlines() if line.strip()]
    headline = lines[0] if lines else "Designed for your next step."
    script = " ".join(lines[1:]) if len(lines) > 1 else raw_output
    return headline[:120], script


@st.cache_resource(show_spinner=False)
def _load_sentence_transformer(model_id: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id)


@st.cache_resource(show_spinner=False)
def _load_clip_components(model_id: str):
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner=False)
def _load_text2text_components(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return tokenizer, model


@st.cache_resource(show_spinner=False)
def _load_hf_inference_client(model_id: str, token: str):
    return cast(Any, InferenceClient(model=model_id, token=token))


class HFProfileEncoder(ProfileEncoder):
    """Pipeline 1: profile text to embedding vector."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or config.PROFILE_EMBEDDING_MODEL_ID

    def encode(self, profile: CustomerProfile) -> EncodedProfile:
        summary = build_profile_summary(profile)
        try:
            model = _load_sentence_transformer(self.model_id)
            embedding = model.encode(summary).tolist()
        except Exception:
            embedding = _deterministic_embedding(summary)
        return EncodedProfile(profile_summary=summary, embedding=embedding)


class HFProductEncoder(ProductEncoder):
    """Pipeline 2: product image+text to embedding vector."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or config.PRODUCT_CLIP_MODEL_ID

    @staticmethod
    def _infer_attributes(product: ProductInfo) -> dict[str, str]:
        baseline = product.attributes.copy() if product.attributes else {}
        category_map = {
            "running": {"support": "smooth transitions", "performance": "distance-ready"},
            "basketball": {"support": "lateral stability", "performance": "explosive cuts"},
            "lifestyle": {"support": "all-day comfort", "performance": "street style"},
            "training": {"support": "multi-direction control", "performance": "gym versatility"},
        }
        for key, value in category_map.items():
            if key in product.category.lower():
                baseline.update(value)
                break
        if "feel" not in baseline:
            baseline["feel"] = "confident and energetic"
        return baseline

    def encode(self, product: ProductInfo) -> EncodedProduct:
        local_image = ensure_local_image(product.image_path_or_url, product.name)
        attributes = self._infer_attributes(product)
        summary = (
            f"{product.name}; category: {product.category}; audience: {product.gender}; "
            f"attributes: {', '.join(f'{k} {v}' for k, v in attributes.items())}."
        )

        try:
            processor, model = _load_clip_components(self.model_id)
            image = Image.open(local_image).convert("RGB")

            processor_any = cast(Any, processor)
            model_any = cast(Any, model)
            text_inputs = processor_any(text=[summary], return_tensors="pt", padding=True, truncation=True)
            image_inputs = processor_any(images=image, return_tensors="pt")

            with torch.no_grad():
                text_features = model_any.get_text_features(**text_inputs)
                image_features = model_any.get_image_features(**image_inputs)

            merged = cast(torch.Tensor, (text_features + image_features) / 2.0)
            embedding = merged[0].cpu().tolist()
        except Exception:
            embedding = _deterministic_embedding(summary, size=512)

        return EncodedProduct(product_summary=summary, embedding=embedding, attributes=attributes)


class HFSloganGenerator(SloganGenerator):
    """Pipeline 3a: profile+product context to short slogan."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or config.SLOGAN_GENERATION_MODEL_ID

    def generate(
        self,
        profile: CustomerProfile,
        encoded_profile: EncodedProfile,
        product: ProductInfo,
        encoded_product: EncodedProduct,
    ) -> str:
        tokenizer, model = _load_text2text_components(self.model_id)
        prompt = build_slogan_prompt(
            profile_summary=encoded_profile.profile_summary,
            product=product,
            product_attributes=encoded_product.attributes,
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=True,
                temperature=0.8,
            )
        raw = tokenizer.decode(outputs[0], skip_special_tokens=True) if len(outputs) else ""
        return parse_generated_slogan(raw)


class HFScriptGenerator(ScriptGenerator):
    """Pipeline 3b: profile+product context to headline and long-form script."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or config.SCRIPT_GENERATION_MODEL_ID

    def generate(
        self,
        profile: CustomerProfile,
        encoded_profile: EncodedProfile,
        product: ProductInfo,
        encoded_product: EncodedProduct,
        slogan: str,
    ) -> MarketingAssets:
        tokenizer, model = _load_text2text_components(self.model_id)
        prompt = build_script_prompt(
            profile_summary=encoded_profile.profile_summary,
            slogan=slogan,
            product=product,
            product_attributes=encoded_product.attributes,
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=True,
                temperature=0.8,
            )
        raw = tokenizer.decode(outputs[0], skip_special_tokens=True) if len(outputs) else ""
        headline, script = parse_generated_script(raw)

        debug = {
            "model_id": self.model_id,
            "raw_text_output": raw,
            "profile_embedding_size": len(encoded_profile.embedding),
            "product_embedding_size": len(encoded_product.embedding),
            "profile": asdict(profile),
        }
        return MarketingAssets(slogan=slogan, headline=headline, script=script, debug_metadata=debug)


class HFVideoGenerator(VideoGenerator):
    """Pipeline 4: text+image to promotional short video, with robust fallback."""

    def __init__(self, model_id: str | None = None, token: str | None = None) -> None:
        self.model_id = model_id or config.VIDEO_MODEL_ID
        self.token = token or config.HF_TOKEN

    @staticmethod
    def _build_video_prompt(profile: CustomerProfile, product: ProductInfo, assets: MarketingAssets) -> str:
        city = profile.nationality or "a modern city"
        age_group = _age_group(profile.age)
        return (
            f"A synthetic {age_group} {profile.gender} from {city} wearing {product.name} Nike shoes, "
            "smiling and walking or lightly jogging through a modern city street at golden hour, "
            "focus on the shoes and natural body movement, realistic, high-quality, cinematic lighting, "
            "shallow depth of field, 35mm lens, natural colors, smooth camera movement, premium sportswear ad, "
            "480p video, no logos, no celebrity likeness, synthetic person only."
        )

    def _try_inference_api(self, prompt: str) -> bytes | None:
        if not config.USE_HF_INFERENCE_API_FOR_VIDEO or not self.token:
            return None

        try:
            client = _load_hf_inference_client(self.model_id, self.token)
            if hasattr(client, "text_to_video"):
                result = client.text_to_video(prompt)
                if isinstance(result, bytes):
                    return result
            return None
        except Exception:
            return None

    def generate(self, profile: CustomerProfile, product: ProductInfo, assets: MarketingAssets) -> str:
        prompt = self._build_video_prompt(profile, product, assets)
        output_stem = sanitize_filename(f"{profile.name}-{product.product_id}-ad")
        final_slogan_text = assets.final_slogan_text or f"{assets.slogan}, {profile.name}"

        video_bytes = self._try_inference_api(prompt)
        if video_bytes:
            raw_video_path = save_video_bytes(video_bytes, prefix=f"{output_stem}-raw")
            return compose_final_ad_video(
                source_video_path=raw_video_path,
                output_stem=output_stem,
                final_slogan_text=final_slogan_text,
            )

        # Streamlit Cloud-safe fallback when inference or local TI2V is unavailable.
        return create_fallback_banner_video(
            image_path_or_url=product.image_path_or_url,
            slogan=assets.slogan,
            headline=assets.headline,
            output_stem=output_stem,
            final_slogan_text=final_slogan_text,
        )
