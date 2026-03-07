# AI Storyboard and Animation Studio

AI Storyboard and Animation Studio is a Streamlit-based production workspace for turning a topic brief into a structured creative package. The intended workflow starts with a topic and supporting description, refines that material into scene-level narrative units, prepares storyboard assets for an Excel workbook (分鏡), and ultimately feeds a final animation deliverable with dialogue, sound design, ambience, and music.

The current repository ships with mock pipeline components so the end-to-end flow can be exercised without external model providers or media-generation services.

## Workflow Overview

The application is organized around four production steps:

1. Topic Intake
	Enter a topic plus several sentences of description that define tone, intent, structure, and visual direction.
2. Narrative Refinement
	Convert the input brief into clearer scene breakdowns and script-ready planning material.
3. Storyboard Workbook Preparation
	Generate storyboard-supporting assets intended for an Excel workbook containing frame descriptions, sketches, and script notes.
4. Final Animation Delivery
	Render or package the final animation output, including sound-related layers such as dialogue, effects, ambience, and music.

## Current Implementation Scope

This repository currently provides a professional UI shell and mock backend behavior for the full production flow.

- The refiner mock splits input text into scene-style segments.
- The storyboard mock returns placeholder sketch/image paths.
- The animation mock returns a placeholder video path.
- Excel workbook generation and real audio or animation rendering are not implemented yet.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application opens in your browser at `http://localhost:8501`.

If you plan to integrate real AI providers, create a `.env` file for the relevant API keys before launching the app.

## Repository Structure

```text
app.py                  Streamlit application and UI workflow
interfaces.py           Abstract interfaces for each pipeline stage
mock_implementations.py Mock backends for local testing and demos
config.yaml             Default pipeline configuration
requirements.txt        Python dependencies
```

## Pipeline Interfaces

| Stage | Interface | Purpose |
|---|---|---|
| Refiner | `ScriptRefiner` | Converts topic input into structured scene descriptions |
| Storyboard | `StoryboardGenerator` | Produces sketch or frame assets for storyboard preparation |
| Animation | `VideoGenerator` | Produces the final animation or preview output |

## Extending the Pipeline

To integrate real services:

1. Create a new implementation class that inherits from the appropriate interface in `interfaces.py`.
2. Register the new class in the corresponding model registry in `app.py`.
3. Select the implementation from the sidebar in the Streamlit application.

## Recommended Next Steps

For a production-ready version of this project, the next logical additions are:

1. Export a real Excel storyboard workbook with columns for scene number, script, description, sketch reference, and shot notes.
2. Replace the storyboard mock with an image or sketch generation backend.
3. Replace the animation mock with a renderer or orchestration layer for video, voice, sound effects, and music.