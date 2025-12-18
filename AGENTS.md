# Repository Guidelines

## Project Structure & Module Organization
- `cm1_gf2_gf7.py`: Annotation-driven flow for scanning TIFFs, matching shapefile labels, and optionally cropping/tiling exports; set `SHP_FILE_PATH`, `DATA_FOLDER_PATH`, `OUTPUT_FOLDER_PATH`, and `PROCESSING_MODE` near the top.
- `process_tile_fusion.py`: Tiling plus pan-sharpening pipeline; adjust SHP/DATA/OUTPUT paths, overlap, and tiling strategy constants in the header.
- `processing_common.py`: Shared helpers for folder scanning, PAN/MSS pairing, tiling logic, annotation matching, and stats.
- `image_utils.py`: RPC parsing, coordinate transforms, normalization, and pan-sharpen helpers reused across pipelines.
- Data I/O expects absolute Windows paths; keep raw imagery and shapefiles outside the repo and point config constants to them.

## Build, Test, and Development Commands
- Create venv: `python -m venv .venv` then activate with `.\\.venv\\Scripts\\activate`.
- Install deps: `pip install geopandas rasterio shapely pyproj pillow opencv-python numpy`.
- Run annotation/crop/tile flow: `python cm1_gf2_gf7.py` after setting path constants and `PROCESSING_MODE`.
- Run tiling + fusion flow: `python process_tile_fusion.py` with your tiling strategy and output paths configured; prefer SSD storage for large imagery.
- Scripts set `OPENCV_IO_MAX_IMAGE_PIXELS` and emit progress logs; watch console output for rasterio/OpenCV warnings.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation; use `snake_case` for functions/variables and `UPPER_SNAKE` for config constants.
- Keep docstrings/comments concise; preserve existing English/Chinese notes and logging tone.
- Favor pure helpers in `image_utils` and `processing_common`; avoid implicit global state beyond the explicit config blocks at the top of scripts.

## Testing Guidelines
- No automated tests yet; sanity-check on a tiny sample folder and verify expected TIFF/JSON counts and paths.
- For new geometry/tiling utilities, lightweight `pytest` cases are welcome; keep fixtures outside the repo.
- Inspect generated `*_info.json` metadata; confirm logs show expected tile and annotation stats.

## Commit & Pull Request Guidelines
- Use descriptive, imperative commit subjects (e.g., "Add RPC parsing guard", "Refine tile fusion overlap"); note data expectations in the body when relevant.
- Do not commit raw imagery, shapefiles, or large intermediates; add ignores as needed.
- PRs should state which pipeline you touched, before/after behavior, sample command/config used, and validation notes (logs, counts, or screenshots).

## Security & Configuration Tips
- Paths default to absolutes; prefer project-local overrides or environment variables when sharing configurations.
- Confirm output directories before runs; overlapping paths can overwrite results, so back up original imagery.
- Never hardcode credentials; keep raw datasets outside the repo and reference via absolute paths.
