# Smart Nike Shoe Ad Studio

Personalized Nike Shoe Video Ads with Hugging Face multi-modal pipelines.

This Streamlit app generates a personalized Nike-style marketing video experience from a customer profile:
- Name, Age, Gender, Nationality, Language, Product
- Optional customer notes

The UI mimics a Nike online storefront and replaces the hero banner with a generated promotional video when the user clicks Generate Marketing Video.

## Business Objective

Smart Nike Shoe Ad Studio demonstrates how profile-level personalization can be transformed into targeted, product-specific ad creative:
- The customer profile drives personalization intent.
- A curated Nike shoe catalog anchors product context.
- Hugging Face pipelines generate ad copy and video assets.

This project is designed for:
- ISOM5240 Deep Learning coursework (Hugging Face-only model stack)
- Portfolio-quality software architecture
- Future production extensibility

## Pipeline Architecture

Core flow (4 stages):
1. User Profile Embedding
2. Product Understanding / Embedding
3. Personalized Marketing Script and Slogan
4. Image/Text to Video Generation (with robust fallback banner video)

Logical diagram:

```text
Customer Profile + Notes
				|
				v
[Pipeline 1] Profile Encoder (Sentence Embeddings)
				|
				v
Selected Nike Product (name/category/image/url)
				|
				v
[Pipeline 2] Product Encoder (CLIP text+image)
				|
				v
[Pipeline 3] Copy Generator (FLAN-T5 or fine-tuned variant)
				|
				v
[Pipeline 4] Video Generator (HF Inference API or fallback video)
				|
				v
Nike-style storefront hero banner video + slogan + script
```

## Hugging Face Models

Default model ids are configurable in config.py.

- sentence-transformers/all-MiniLM-L6-v2
	- Role: profile embedding (Pipeline 1)
	- Link: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

- openai/clip-vit-base-patch32
	- Role: product text+image embedding (Pipeline 2)
	- Link: https://huggingface.co/openai/clip-vit-base-patch32

- google/flan-t5-base
	- Role: slogan/headline/script generation (Pipeline 3)
	- Link: https://huggingface.co/google/flan-t5-base

- Wan-AI/Wan2.2-TI2V-5B-Diffusers
	- Role: promotional video generation target model (Pipeline 4)
	- Link: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers

## Fine-Tuned Model Hook

Pipeline 3 is designed to be replaced by your own fine-tuned marketing model.

Where to plug in:
- Update COPY_GENERATION_MODEL_ID in config.py
- Optionally set environment variable COPY_GENERATION_MODEL_ID in .env

Example target model id:
- my-username/nike-marketing-flan-t5-base

## Project Structure

```text
app.py                  Streamlit UI and pipeline orchestration
interfaces.py           Dataclasses, protocols, and stage registries
hf_pipelines.py         Hugging Face implementations for all 4 stages
mock_implementations.py Deterministic no-network fallback implementations
nike_catalog.py         Curated Nike shoe catalog helpers
media_utils.py          Video/image utility helpers and fallback banner video
config.py               Model IDs, tokens, feature flags, artifact paths
requirements.txt        Python dependencies
```

## Run Locally

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Create .env from .env.example and add Hugging Face token

```bash
HF_TOKEN=your_huggingface_token_here
```

3. Launch app

```bash
streamlit run app.py
```

## Streamlit Cloud Notes

This app is optimized for constrained environments:
- Heavy video generation defaults to Hugging Face Inference API mode when token is available.
- If token/video inference is unavailable, app falls back to a generated static-banner MP4 so the full flow still works.
- Cache usage:
	- @st.cache_data for catalog and deterministic image prep
	- @st.cache_resource for model/client loaders

## UI Behavior

- Sidebar inputs:
	- Name, Age, Gender, Nationality, Language, Product, Additional notes
- Action:
	- Generate Marketing Video
- Main panel:
	- Nike-style storefront layout
	- Hero banner becomes generated video
	- Displays personalized slogan, headline, and script

## Limitations

- Real TI2V API behavior can vary by hosted endpoint and account capability.
- Inference latency depends on model availability and queue time.
- Current text generation enforces English output for consistency.
- Catalog is curated demo data, not a live Nike commerce feed.

## Future Work

1. Add multilingual copy generation and language-specific model routing.
2. Enable optional TTS voice-over and audio-video muxing.
3. Add A/B ad prompt experimentation and conversion analytics hooks.
4. Expand product coverage and dynamic catalog ingestion pipeline.