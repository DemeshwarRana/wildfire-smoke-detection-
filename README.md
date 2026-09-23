# Wildfire Smoke Detection — Early-Warning Image Classification

A 3-class image classifier (transfer learning, TensorFlow/Keras) that distinguishes **smoke**, **fire**, and **non-fire** images from forest/outdoor camera photos.

The business problem this mirrors: fire agencies and utility companies run networks of fixed cameras and need to catch **smoke** — before it becomes an active fire — early enough to dispatch a crew. A model that only detects flames is too late; the value is in the smoke-only class. That's why this is framed as 3-class (smoke / fire / non-fire) rather than a simple fire/no-fire binary, and why the evaluation section looks specifically at how often "smoke" gets confused with "non-fire" (a missed early warning) vs. with "fire" (a less costly mix-up).

## What's in this repo

- `wildfire_smoke_detection.ipynb` — the full pipeline: EDA → data pipeline → model → training → evaluation → business-framed discussion of errors.
- `requirements.txt` — Python dependencies.

The dataset itself is **not** committed here (see below).

## Setup

```bash
git clone https://github.com/DemeshwarRana/wildfire-smoke-detection.git
cd wildfire-smoke-detection
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Get the dataset

Dataset: **[Forest Fire, Smoke, and Non-Fire Image Dataset](https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset)** (Kaggle, ~6 GB).

**Option A — run on Kaggle (recommended, no local download):**

1. Open the dataset page above and click **"New Notebook"** — this creates a Kaggle-hosted notebook with the dataset already mounted at `/kaggle/input/forest-fire-smoke-and-non-fire-image-dataset/`.
2. In that notebook, **File → Upload Notebook** and upload `wildfire_smoke_detection.ipynb` from this repo.
3. **Settings → Accelerator → GPU T4 x2** (free tier).
4. Run all cells — the notebook auto-detects the Kaggle path, no local disk used at all.
5. When finished, **File → Download** to bring the executed notebook (with results) back here.

**Option B — fetch it via the Kaggle API from your local notebook (no manual browser download):**

1. Go to kaggle.com → profile picture → **Account** → **API** → **"Create New Token"**. This downloads `kaggle.json`.
2. Place it at `C:\Users\ragha\.kaggle\kaggle.json` (folder already created in this repo's setup).
3. `pip install -r requirements.txt` (includes `kagglehub`).
4. Just run the notebook — the data cell calls `kagglehub.dataset_download(...)` automatically if `data/train` isn't present, fetching straight from Kaggle's API instead of a manual zip download. It's cached locally after the first run (default: `C:\Users\ragha\.cache\kagglehub\`), so this still uses ~6GB of local disk, just without you manually clicking through the website.

**Option C — download locally by hand:**

1. Download and extract the dataset into `data/` so the repo looks like this (the notebook auto-detects class names from the subfolders, so it doesn't matter if the exact folder names differ slightly from below — just make sure train/test each have one subfolder per class):

```
wildfire-smoke-detection/
├── wildfire_smoke_detection.ipynb
└── data/
    ├── train/
    │   ├── fire/
    │   ├── smoke/
    │   └── non-fire/
    └── test/
        ├── fire/
        ├── smoke/
        └── non-fire/
```

2. Run:

```bash
jupyter notebook wildfire_smoke_detection.ipynb
```

Run the cells in order from the repo root so the relative `data/...` paths resolve correctly.

## Key result

_To be filled in after training — see Evaluation section of the notebook._

## References

- Amerzish Minha. *Forest Fire, Smoke, and Non-Fire Image Dataset*. Kaggle.
