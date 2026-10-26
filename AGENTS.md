# Repository Guidelines

## Project Structure & Module Organization

FloodDiffusion generates streaming human motion from text using a VAE and latent diffusion forcing. Root scripts handle training, pretokenization, generation, and visualization. `models/` contains networks, `datasets/` contains HumanML3D and BABEL loaders, `metrics/` implements evaluation, and `utils/` provides shared helpers. Experiment settings live in `configs/`. `web_demo/` contains the Flask app, templates, and JavaScript/CSS assets. Downloaded dependencies, datasets, and checkpoints belong in `deps/`, `raw_data/`, and `outputs/`.

## Build, Test, and Development Commands

Run Python scripts from the repository root. Use Python 3.10 and follow the README for CUDA, PyTorch, and Flash Attention setup. No separate build step is required.

- `pip install -r requirements.txt`: install core dependencies.
- `cp configs/paths_default.yaml configs/paths.yaml`: initialize local configuration; edit paths and WandB settings.
- `pip install huggingface_hub` then `python download_assets.py --with-dataset`: download dependencies, checkpoints, and datasets.
- `python train_vae.py --config configs/vae_wan_1d.yaml --override train=True`: train the VAE.
- `python pretokenize_vae.py --config configs/vae_wan_1d.yaml`: cache motion tokens.
- `python train_ldf.py --config configs/ldf.yaml --override train=True`: train diffusion on HumanML3D.
- `python train_ldf.py --config configs/ldf.yaml`: evaluate the configured checkpoint.
- `python generate_ldf.py --config configs/stream.yaml`: run interactive generation.
- `pip install -r web_demo/requirements.txt` then `cd web_demo && bash server.sh start`: launch the demo on port 5000.

## Coding Style & Naming Conventions

Follow surrounding code: four-space Python indentation, `snake_case` functions/modules, and `PascalCase` classes. Keep experiment options in YAML and reuse configuration helpers in `utils/initialize.py`. No formatter or linter configuration is checked in; avoid unrelated formatting changes.

## Testing Guidelines

No dedicated automated test suite or coverage threshold is present. Validate model changes with the relevant evaluation script and a short generation run using prepared assets. Record configuration, checkpoint, seed, and metrics. For demo changes, check prompt updates and motion playback.

## Commit & Pull Request Guidelines

History uses short, descriptive subjects without consistent prefixes. Use imperative summaries such as `Fix streaming dropout`. Keep changes focused. PRs should explain behavior changes, link relevant issues, and report validation commands, results, and hardware limitations. Include screenshots for visual changes.

## Configuration & Assets

Keep credentials in ignored `configs/paths.yaml`. Do not commit datasets, checkpoints, outputs, or WandB logs. Preserve attribution in `THIRD_PARTY_LICENSES.md`.
