"""Deterministic mock backends for Smart Nike Shoe Ad Studio."""

from __future__ import annotations

import hashlib

from interfaces import (
    CopyGenerator,
    CustomerProfile,
    EncodedProduct,
    EncodedProfile,
    MarketingAssets,
    ProductEncoder,
    ProductInfo,
    ProfileEncoder,
    VideoGenerator,
)
from media_utils import create_fallback_banner_video, sanitize_filename


def _mock_embedding(key: str, size: int) -> list[float]:
    seed = hashlib.sha1(key.encode("utf-8")).digest()
    values: list[float] = []
    while len(values) < size:
        for byte in seed:
            values.append((byte / 255.0) * 2.0 - 1.0)
            if len(values) == size:
                break
        seed = hashlib.sha1(seed).digest()
    return values


class MockProfileEncoder(ProfileEncoder):
    def encode(self, profile: CustomerProfile) -> EncodedProfile:
        summary = (
            f"Name: {profile.name}; Age: {profile.age}; Gender: {profile.gender}; "
            f"Nationality: {profile.nationality or 'Not specified'}; Language: {profile.language}; "
            f"Product ID: {profile.product_id}; Notes: {profile.additional_notes or 'No notes'}."
        )
        return EncodedProfile(profile_summary=summary, embedding=_mock_embedding(summary, 384))


class MockProductEncoder(ProductEncoder):
    def encode(self, product: ProductInfo) -> EncodedProduct:
        attributes = (product.attributes or {}).copy()
        if "benefit" not in attributes:
            attributes["benefit"] = "all-day comfort"
        summary = f"{product.name} in {product.category} for {product.gender}."
        return EncodedProduct(
            product_summary=summary,
            embedding=_mock_embedding(f"{product.product_id}:{summary}", 512),
            attributes=attributes,
        )


class MockCopyGenerator(CopyGenerator):
    def generate(
        self,
        profile: CustomerProfile,
        encoded_profile: EncodedProfile,
        product: ProductInfo,
        encoded_product: EncodedProduct,
    ) -> MarketingAssets:
        slogan = f"{profile.name}, Own Every Step"
        headline = f"{product.name} built for your {product.category.lower()} rhythm"
        script = (
            f"Meet {profile.name}'s personalized Nike moment. "
            f"The {product.name} blends {encoded_product.attributes.get('benefit', 'performance support')} "
            f"with a style that fits a {profile.age}-year-old {profile.gender.lower()} customer. "
            "Every stride feels lighter, more confident, and ready for daily wins. "
            "Lace up, move happy, and turn your routine into a statement."
        )
        return MarketingAssets(
            slogan=slogan,
            headline=headline,
            script=script,
            debug_metadata={"mode": "mock", "profile_summary": encoded_profile.profile_summary},
        )


class MockVideoGenerator(VideoGenerator):
    def generate(self, profile: CustomerProfile, product: ProductInfo, assets: MarketingAssets) -> str:
        stem = sanitize_filename(f"mock-{profile.name}-{product.product_id}")
        return create_fallback_banner_video(
            image_path_or_url=product.image_path_or_url,
            slogan=assets.slogan,
            headline=assets.headline,
            output_stem=stem,
        )
