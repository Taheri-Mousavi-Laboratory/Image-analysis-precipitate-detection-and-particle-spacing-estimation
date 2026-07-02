# Image-analysis-precipitate-detection-and-particle-spacing-estimation
# Precipitate & Inter-Particle Spacing Analysis

Image-analysis pipelines that detect precipitates in SEM/TEM micrographs and compute two quantitative descriptors:

- **Precipitate size** — equivalent diameter, *d*
- **Inter-particle spacing** — nearest-neighbour distance, *s*

Two scripts are included, one per heat-treatment condition. Both scripts are
fully self-contained: load image(s) → detect & validate precipitates →
compute statistics → print results → show summary plot(s).

## What's in this repo

| Script | Condition | Input | What it does |
|---|---|---|---|
| `src/precipitate_analysis_725C_1hr.py` | 725 °C, 1 hr | 1 image | Example file showing the detection of precipitates, prints diameter/spacing stats, shows 1 summary figure |
| `src/precipitate_analysis_725C_5hr.py` | 725 °C, 5 hr | 3 images (ROI-1/2/3) | Runs the same analysis on each of the 3 sub-region images, then **pools** all three into combined statistics (printed only, no extra plot) |

Both scripts report diameter and spacing in px, µm, and nm. This pipeline is scoped to size and spacing only.

## Repository layout

```
precipitate-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                                 <- put your .tif micrographs here (not tracked by git)
│   ├── 1_0HR_725C.tif
│   ├── 5_0HR_725C_10.tif
│   ├── 5_0HR_725C_20.tif
│   └── 5_0HR_725C_30.tif
├── outputs/                              <- save exported figures/CSVs here if desired (not tracked by git)
└── src/
    ├── precipitate_analysis_725C_1hr.py
    └── precipitate_analysis_725C_5hr.py
```

Keeping raw micrographs in `data/` (git-ignored) and code in `src/` keeps the
repo lightweight and avoids committing large/proprietary image files.

## Setup

```bash
git clone <your-repo-url>
cd precipitate-analysis
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Drop your micrograph(s) into `data/` using the filenames shown in the
   layout above (or edit the paths in the config section of each script —
   see below).
2. Run the script for your condition:

```bash
python src/precipitate_analysis_725C_1hr.py
python src/precipitate_analysis_725C_5hr.py
```

Or open either file in Jupyter and run all cells — both are written as plain
top-to-bottom scripts with an `if __name__ == "__main__":` block, so they work
the same way in a notebook (just run the whole file / paste into cells).

## Configuring for your own images

Each script has a **USER CONFIGURATION** block near the top:

```python
IMAGE_PATH = "data/1_0HR_725C.tif"
SCALE_BAR_MICRONS = 1.0     # length of your scale bar, in micrometres
SCALE_BAR_PIXELS  = 129     # length of that same scale bar, in pixels
EDGE_TO_EDGE      = False   # True = edge-to-edge spacing, False = centre-to-centre
```

Update `SCALE_BAR_MICRONS` / `SCALE_BAR_PIXELS` to match your image's scale
bar — everything else (nm/px conversion, all reported values) derives from
this. The 5 hr script has the equivalent `IMAGE_PATHS` dict for its 3 ROIs.

Detection sensitivity (blob size range, brightness/contrast thresholds, shape
filters) is set in `detect_blobs()` and `validate_precipitates()`. The two
scripts use different default tunings because 5 hr precipitates are larger
and brighter than 1 hr ones — inline comments in `precipitate_analysis_725C_5hr.py`
explain which knob to adjust if detection is picking up too much or too
little.

## What each run produces

**1 hr script** — one run:
1. Console: candidate/validated particle counts, then a statistics block
   (mean, std dev, median, min, max) for diameter and spacing, in px/µm/nm.
2. One figure: detected-precipitate overlay, diameter histogram, spacing
   histogram, nearest-neighbour connection map, spacing heat-map.

**5 hr script** — one run over 3 images, then a pooled summary:
1. For each ROI: same console statistics block + same one-figure plot as above.
2. After all 3 ROIs: a **pooled** summary — per-ROI breakdown table plus
   combined mean, std dev, median, min, max, CV, SEM, and 95% CI for diameter
   and spacing across all particles from all 3 images. **This step is
   print-only — no plot is generated.**

## Method notes

- Precipitates are detected via white top-hat filtering + Laplacian-of-
  Gaussian (LoG) blob detection, followed by a brightness/contrast/shape
  validation pass to reject false positives (grain-boundary texture,
  elongated features, etc.).
- Nearest-neighbour spacing is computed with a KD-tree over particle
  centroids; `EDGE_TO_EDGE = True` subtracts both particles' radii from the
  centre-to-centre distance, `False` reports raw centre-to-centre distance.
- All statistics are computed directly from per-particle measurements (not
  from binned histogram data), so mean/median/etc. are exact.
