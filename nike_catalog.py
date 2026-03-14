"""Curated Nike shoe catalog helpers."""

from __future__ import annotations

from interfaces import ProductInfo


_CATALOG: list[dict[str, str]] = [
    {
        "product_id": "pegasus-41-m",
        "name": "Nike Air Zoom Pegasus 41",
        "category": "Running",
        "gender": "Men",
        "image_path_or_url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?auto=format&fit=crop&w=1200&q=80",
        "product_url": "https://www.nike.com.hk/man/shoe/list.htm?intpromo=PNTP",
    },
    {
        "product_id": "vomero-17-f",
        "name": "Nike ZoomX Vomero 17",
        "category": "Running",
        "gender": "Women",
        "image_path_or_url": "https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?auto=format&fit=crop&w=1200&q=80",
        "product_url": "https://www.nike.com.hk/woman/shoe/list.htm?intpromo=PNTP",
    },
    {
        "product_id": "dunk-low-unisex",
        "name": "Nike Dunk Low",
        "category": "Lifestyle",
        "gender": "Unisex",
        "image_path_or_url": "https://images.unsplash.com/photo-1600185365483-26d7a4cc7519?auto=format&fit=crop&w=1200&q=80",
        "product_url": "https://www.nike.com.hk/man/shoe/list.htm?intpromo=PNTP",
    },
    {
        "product_id": "ja-2-basketball",
        "name": "Nike Ja 2",
        "category": "Basketball",
        "gender": "Unisex",
        "image_path_or_url": "https://images.unsplash.com/photo-1514989940723-e8e51635b782?auto=format&fit=crop&w=1200&q=80",
        "product_url": "https://www.nike.com.hk/man/shoe/list.htm?intpromo=PNTP",
    },
    {
        "product_id": "metcon-9-training",
        "name": "Nike Metcon 9",
        "category": "Training",
        "gender": "Unisex",
        "image_path_or_url": "https://images.unsplash.com/photo-1463100099107-aa0980c362e6?auto=format&fit=crop&w=1200&q=80",
        "product_url": "https://www.nike.com.hk/man/shoe/list.htm?intpromo=PNTP",
    },
]


_DEF_ATTRIBUTES = {
    "Running": {"benefit": "responsive cushioning", "feel": "lightweight"},
    "Lifestyle": {"benefit": "all-day comfort", "feel": "street-ready"},
    "Basketball": {"benefit": "court grip", "feel": "explosive"},
    "Training": {"benefit": "stable support", "feel": "versatile"},
}


def _to_product(raw: dict[str, str]) -> ProductInfo:
    category = raw["category"]
    return ProductInfo(
        product_id=raw["product_id"],
        name=raw["name"],
        category=category,
        gender=raw["gender"],
        image_path_or_url=raw["image_path_or_url"],
        product_url=raw["product_url"],
        attributes=_DEF_ATTRIBUTES.get(category, {}).copy(),
    )


def list_products() -> list[ProductInfo]:
    """Return all curated Nike products."""
    return [_to_product(row) for row in _CATALOG]


def get_product(product_id: str) -> ProductInfo:
    """Return a product by id or raise ValueError for invalid ids."""
    for row in _CATALOG:
        if row["product_id"] == product_id:
            return _to_product(row)
    raise ValueError(f"Unknown product_id: {product_id}")


def get_product_by_name(product_name: str) -> ProductInfo:
    """Return a product by display name or raise ValueError when missing."""
    for row in _CATALOG:
        if row["name"] == product_name:
            return _to_product(row)
    raise ValueError(f"Unknown product name: {product_name}")
