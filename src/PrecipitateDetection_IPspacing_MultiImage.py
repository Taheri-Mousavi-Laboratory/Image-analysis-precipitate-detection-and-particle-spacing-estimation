"""
============================================================
 PRECIPITATE & INTER-PARTICLE SPACING ANALYSIS
 Condition : 725 C, 5 hr   (3 sub-region micrographs, pooled)
============================================================
 Workflow
 --------
 1. For each of the 3 sub-images (ROI-1, ROI-2, ROI-3):
      a. Load micrograph
      b. Detect + validate precipitates (top-hat + LoG blobs)
      c. Compute statistical descriptors:
           - Equivalent diameter, d
           - Nearest-neighbour spacing, s
      d. Print descriptors for that image
      e. Show ONE summary figure for that image (detection overlay,
         diameter histogram, spacing histogram, nearest-neighbour
         map, spacing heat-map)

 2. Pool all 3 sub-images into one combined dataset and print POOLED statistics
============================================================
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import filters
from skimage.feature import blob_log
from skimage.measure import label, regionprops
from skimage.morphology import white_tophat, disk
from scipy.spatial import KDTree

# ============================================================
# USER CONFIGURATION
# ============================================================

IMAGE_PATHS = {
    "ROI-1": (
        "data/5_0HR_725C_10.tif"
    ),
    "ROI-2": (
        "data/5_0HR_725C_20.tif"
    ),
    "ROI-3": (
        "data/5_0HR_725C_30.tif"
    ),
}

SCALE_BAR_MICRONS = 1.0    # length of the scale bar, in micrometres
SCALE_BAR_PIXELS  = 129    # length of the same scale bar, in pixels
EDGE_TO_EDGE       = True  # True -> edge-to-edge NN spacing, False -> centre-to-centre


# ============================================================
# STEP 0 -- Load image
# ============================================================

def load_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")

    img_gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    if img_gray_raw.dtype != np.uint8:
        img_gray_raw = cv2.normalize(img_gray_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return img_gray_raw.astype(float) / 255.0


# ============================================================
# STEP 1 -- Pre-processing
# ============================================================

def preprocess(img_gray, tophat_radius=4, smooth_sigma=0.6):
    tophat = white_tophat(img_gray, disk(tophat_radius))
    tophat_norm = tophat / tophat.max() if tophat.max() > 0 else tophat.copy()
    tophat_smooth = filters.gaussian(tophat_norm, sigma=smooth_sigma)
    return tophat_norm, tophat_smooth


# ============================================================
# STEP 2 -- Blob detection (tuned for larger 5 hr precipitates)
# ============================================================

def detect_blobs(
    tophat_smooth,
    min_sigma=1.2, max_sigma=10.0, num_sigma=35,
    threshold=0.04, overlap=0.30,
    min_diameter_px=4, max_diameter_px=50,
):
    blobs = blob_log(
        tophat_smooth,
        min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma,
        threshold=threshold, overlap=overlap,
    )

    df = pd.DataFrame(blobs, columns=["centroid_y", "centroid_x", "sigma"])
    if len(df) == 0:
        print("Blob candidates : 0")
        return df

    df["radius_px"] = np.sqrt(2) * df["sigma"]
    df["diameter_px"] = 2 * df["radius_px"]
    df = df[(df["diameter_px"] >= min_diameter_px) & (df["diameter_px"] <= max_diameter_px)].reset_index(drop=True)

    print(f"Blob candidates : {len(df)}")
    return df


# ============================================================
# STEP 3 -- Validate: dual brightness gate + shape filter
# ============================================================

def validate_precipitates(
    img_gray, tophat_norm, candidates_df,
    local_radius_factor=2.5, threshold_fraction=0.38,
    min_peak_brightness=0.65, min_contrast_ratio=8.0,
    min_area_local=15, max_area_local=9000,
    max_aspect_ratio=1.8, min_circularity=0.50,
    min_solidity=0.72, max_eccentricity=0.80,
    min_fill_fraction=0.15, max_component_count=2,
    mask_radius_factor=1.05,
):
    accepted = []

    if len(candidates_df) == 0:
        final_mask = np.zeros_like(img_gray, dtype=np.uint8)
        return pd.DataFrame(), final_mask, np.zeros_like(img_gray)

    for _, row in candidates_df.iterrows():
        cx, cy = int(round(row["centroid_x"])), int(round(row["centroid_y"]))
        r = int(round(row["radius_px"] * local_radius_factor))
        r_blob = max(2, int(round(row["radius_px"])))
        if r < 3:
            continue

        y1, y2 = max(0, cy - r), min(img_gray.shape[0], cy + r + 1)
        x1, x2 = max(0, cx - r), min(img_gray.shape[1], cx + r + 1)

        patch = tophat_norm[y1:y2, x1:x2]
        if patch.size == 0:
            continue

        yy, xx = np.indices(patch.shape)
        cy_loc, cx_loc = cy - y1, cx - x1
        dist_sq = (xx - cx_loc) ** 2 + (yy - cy_loc) ** 2

        core_mask = dist_sq <= r_blob ** 2
        annulus_mask = (dist_sq > (r_blob * 1.5) ** 2) & (dist_sq <= r ** 2)

        core_vals, annulus_vals = patch[core_mask], patch[annulus_mask]
        if len(core_vals) == 0 or len(annulus_vals) == 0:
            continue

        peak = core_vals.max()
        annulus_mean = annulus_vals.mean()
        if annulus_mean < 1e-6:
            continue
        contrast_ratio = core_vals.mean() / annulus_mean

        if peak < min_peak_brightness or contrast_ratio < min_contrast_ratio:
            continue

        roi_mask = dist_sq <= r ** 2
        vals = patch[roi_mask]
        lo, hi = vals.min(), vals.max()
        if hi <= lo:
            continue

        thresh = lo + threshold_fraction * (hi - lo)
        binary = np.zeros_like(patch, dtype=bool)
        binary[(patch >= thresh) & roi_mask] = True

        labeled = label(binary, connectivity=2)
        props = regionprops(labeled)
        if not props or len(props) > max_component_count:
            continue

        center_lbl = labeled[cy_loc, cx_loc]
        if center_lbl == 0:
            dists = [np.hypot(p.centroid[1] - cx_loc, p.centroid[0] - cy_loc) for p in props]
            sel = props[int(np.argmin(dists))]
        else:
            sel = next((p for p in props if p.label == center_lbl), None)
        if sel is None:
            continue

        area, perim = sel.area, sel.perimeter
        major, minor = sel.axis_major_length, sel.axis_minor_length
        if minor == 0 or perim == 0:
            continue

        aspect_ratio = major / minor
        circularity = 4 * np.pi * area / (perim ** 2)
        solidity = sel.solidity
        eccentricity = sel.eccentricity
        fill_fraction = area / roi_mask.sum()

        if not (
            min_area_local <= area <= max_area_local
            and aspect_ratio <= max_aspect_ratio
            and circularity >= min_circularity
            and solidity >= min_solidity
            and eccentricity <= max_eccentricity
            and fill_fraction >= min_fill_fraction
        ):
            continue

        lcy, lcx = sel.centroid
        accepted.append({
            "centroid_x": x1 + lcx, "centroid_y": y1 + lcy,
            "radius_px_log": row["radius_px"], "diameter_px_log": row["diameter_px"],
            "area_local": area, "equivalent_diameter_px": sel.equivalent_diameter_area,
            "axis_major_length": major, "axis_minor_length": minor,
            "aspect_ratio": aspect_ratio, "circularity": circularity,
            "solidity": solidity, "eccentricity": eccentricity,
            "fill_fraction": fill_fraction, "peak_brightness": peak,
            "contrast_ratio": contrast_ratio,
        })

    vdf = pd.DataFrame(accepted)
    print(f"After dual brightness + shape filter : {len(vdf)}")

    final_mask = np.zeros_like(img_gray, dtype=np.uint8)
    yy, xx = np.indices(img_gray.shape)
    for _, r in vdf.iterrows():
        rad = max(1.5, (r["equivalent_diameter_px"] / 2) * mask_radius_factor)
        final_mask[(xx - r["centroid_x"]) ** 2 + (yy - r["centroid_y"]) ** 2 <= rad ** 2] = 255

    extracted = np.zeros_like(img_gray)
    extracted[final_mask > 0] = img_gray[final_mask > 0]

    return vdf, final_mask, extracted


# ============================================================
# STEP 4 -- Statistical descriptors (per image)
# ============================================================

def diameter_stats(validated_df, mpp):
    d_px = validated_df["equivalent_diameter_px"].values
    mean_um, std_um = float(np.mean(d_px) * mpp), float(np.std(d_px, ddof=1) * mpp)
    median_um = float(np.median(d_px) * mpp)
    min_um, max_um = float(np.min(d_px) * mpp), float(np.max(d_px) * mpp)

    return {
        "mean_px": float(np.mean(d_px)), "std_px": float(np.std(d_px, ddof=1)),
        "median_px": float(np.median(d_px)), "min_px": float(np.min(d_px)), "max_px": float(np.max(d_px)),
        "mean_um": mean_um, "std_um": std_um, "median_um": median_um, "min_um": min_um, "max_um": max_um,
        "mean_nm": mean_um * 1000, "std_nm": std_um * 1000, "median_nm": median_um * 1000,
        "min_nm": min_um * 1000, "max_nm": max_um * 1000,
        "n": len(d_px),
    }


def spacing_stats(validated_df, mpp, edge_to_edge):
    coords = validated_df[["centroid_x", "centroid_y"]].values
    radii = validated_df["equivalent_diameter_px"].values / 2.0

    tree = KDTree(coords)
    nn_dists, nn_idx = tree.query(coords, k=2)
    nn_dist, nn_index = nn_dists[:, 1], nn_idx[:, 1]

    if edge_to_edge:
        nn_spacing = np.maximum(nn_dist - radii - radii[nn_index], 0.0)
    else:
        nn_spacing = nn_dist.copy()

    mean_um, std_um = float(np.mean(nn_spacing) * mpp), float(np.std(nn_spacing, ddof=1) * mpp)
    median_um = float(np.median(nn_spacing) * mpp)
    min_um, max_um = float(np.min(nn_spacing) * mpp), float(np.max(nn_spacing) * mpp)

    stats = {
        "mean_px": float(np.mean(nn_spacing)), "std_px": float(np.std(nn_spacing, ddof=1)),
        "median_px": float(np.median(nn_spacing)), "min_px": float(np.min(nn_spacing)), "max_px": float(np.max(nn_spacing)),
        "mean_um": mean_um, "std_um": std_um, "median_um": median_um, "min_um": min_um, "max_um": max_um,
        "mean_nm": mean_um * 1000, "std_nm": std_um * 1000, "median_nm": median_um * 1000,
        "min_nm": min_um * 1000, "max_nm": max_um * 1000,
        "edge_to_edge": edge_to_edge, "n": len(nn_spacing),
    }
    return nn_spacing, nn_index, stats


# ============================================================
# STEP 5 -- Per-image summary plot (5 panels, no area-fraction panel)
# ============================================================

def plot_summary(img_gray, validated_df, result_df, diam_stats, spacing_stats, roi_label):
    spacing_label = "Edge-to-edge spacing, s (nm)" if spacing_stats["edge_to_edge"] else "Centre-to-centre distance, s (nm)"

    fig = plt.figure(figsize=(19, 12))
    gs = fig.add_gridspec(2, 6, hspace=0.45, wspace=1.1)

    ax1 = fig.add_subplot(gs[0, 0:2])   # detected precipitates overlay
    ax2 = fig.add_subplot(gs[0, 2:4])   # diameter histogram
    ax3 = fig.add_subplot(gs[0, 4:6])   # spacing histogram
    ax4 = fig.add_subplot(gs[1, 1:3])   # nearest-neighbour connections
    ax5 = fig.add_subplot(gs[1, 3:5])   # spacing heat-map

    fig.suptitle(
        f"Precipitate Analysis  |  Condition: 725 C, 5 hr -- {roi_label}  |  Sample size n = {diam_stats['n']}",
        fontsize=15, fontweight="bold", y=0.98,
    )

    # 1 -- detected precipitates overlay
    ax1.imshow(img_gray, cmap="gray")
    for _, row in validated_df.iterrows():
        r = max(2.2, row["equivalent_diameter_px"] / 2)
        ax1.add_patch(plt.Circle((row["centroid_x"], row["centroid_y"]), r,
                                  fill=False, color="cyan", linewidth=0.7, alpha=0.85))
    ax1.set_title(f"Detected Precipitates (n = {diam_stats['n']})", fontsize=11, pad=10)
    ax1.axis("off")

    # 2 -- diameter histogram (nm)
    d_nm = result_df["equivalent_diameter_nm"]
    ax2.hist(d_nm, bins=max(10, diam_stats["n"] // 15), color="steelblue", edgecolor="white", linewidth=0.5)
    ax2.axvline(diam_stats["mean_nm"], color="red", lw=1.8, linestyle="--",
                label=f"Mean = {diam_stats['mean_nm']:.2f} nm")
    ax2.axvline(diam_stats["median_nm"], color="orange", lw=1.8, linestyle=":",
                label=f"Median = {diam_stats['median_nm']:.2f} nm")
    ax2.set_xlabel("Equivalent Diameter, d (nm)", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_title("Precipitate Size Distribution", fontsize=11, pad=10)
    ax2.legend(fontsize=8, loc="upper right", frameon=True)
    ax2.tick_params(labelsize=9)

    # 3 -- spacing histogram (nm)
    s_nm = result_df["nn_spacing_nm"]
    ax3.hist(s_nm, bins=max(10, len(s_nm) // 15), color="darkorange", edgecolor="white", linewidth=0.5)
    ax3.axvline(spacing_stats["mean_nm"], color="red", lw=1.8, linestyle="--",
                label=f"Mean = {spacing_stats['mean_nm']:.2f} nm")
    ax3.axvline(spacing_stats["median_nm"], color="blue", lw=1.8, linestyle=":",
                label=f"Median = {spacing_stats['median_nm']:.2f} nm")
    ax3.set_xlabel(spacing_label, fontsize=10)
    ax3.set_ylabel("Count", fontsize=10)
    ax3.set_title("Inter-Particle Spacing Distribution", fontsize=11, pad=10)
    ax3.legend(fontsize=8, loc="upper right", frameon=True)
    ax3.tick_params(labelsize=9)

    # 4 -- nearest-neighbour connections
    ax4.imshow(img_gray, cmap="gray")
    for i, row in result_df.iterrows():
        j = int(row["nn_index"])
        x0, y0 = row["centroid_x"], row["centroid_y"]
        x1, y1 = result_df.loc[j, "centroid_x"], result_df.loc[j, "centroid_y"]
        ax4.plot([x0, x1], [y0, y1], color="cyan", lw=0.6, alpha=0.5)
        ax4.plot(x0, y0, "o", color="yellow", markersize=2, alpha=0.8)
    ax4.set_title("Nearest-Neighbour Connections", fontsize=11, pad=10)
    ax4.axis("off")

    # 5 -- spacing heat-map
    ax5.imshow(img_gray, cmap="gray", alpha=0.5)
    sc = ax5.scatter(result_df["centroid_x"], result_df["centroid_y"], c=result_df["nn_spacing_nm"],
                      cmap="plasma", s=18, linewidths=0.4, edgecolors="white", alpha=0.9)
    cbar = plt.colorbar(sc, ax=ax5, fraction=0.045, pad=0.03)
    cbar.set_label(spacing_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax5.set_title("Spacing Map", fontsize=11, pad=10)
    ax5.axis("off")

    plt.subplots_adjust(top=0.90, bottom=0.05, left=0.03, right=0.98)
    plt.show()


# ============================================================
# STEP 6 -- Per-image pretty printer
# ============================================================

def print_summary(diam_stats, spacing_stats, mpp, npp, roi_label):
    spacing_type = "edge-to-edge" if spacing_stats["edge_to_edge"] else "centre-to-centre"

    d = diam_stats
    s = spacing_stats

    line = "=" * 62
    print("")
    print(line)
    print("  PRECIPITATE ANALYSIS SUMMARY  --  725 C, 5 hr -- {}".format(roi_label))
    print("  Scale: {:.6f} um/px  =  {:.2f} nm/px".format(mpp, npp))
    print(line)

    print("")
    print("  PARTICLE DIAMETER, d   (n = {})".format(d["n"]))
    print("  {:<18}: {:>8.2f} px   {:>10.2f} nm".format("Mean", d["mean_px"], d["mean_nm"]))
    print("  {:<18}: {:>8.2f} px   {:>10.2f} nm".format("Std Dev", d["std_px"], d["std_nm"]))
    print("  {:<18}: {:>8.2f} px   {:>10.2f} nm".format("Median", d["median_px"], d["median_nm"]))
    print("  {:<18}: {:>8.2f} px   {:>10.2f} nm".format("Min", d["min_px"], d["min_nm"]))
    print("  {:<18}: {:>8.2f} px   {:>10.2f} nm".format("Max", d["max_px"], d["max_nm"]))

    print("")
    print("  NEAREST-NEIGHBOUR SPACING, s  -- {}  (n = {})".format(spacing_type, s["n"]))
    print("  {:<18}: {:>8.2f} px   {:>10.2f} nm".format("Mean", s["mean_px"], s["mean_nm"]))
    print("  {:<18}: {:>8.2f} px   {:>10.2f} nm".format("Std Dev", s["std_px"], s["std_nm"]))
    print("  {:<18}: {:>8.2f} px   {:>10.2f} nm".format("Median", s["median_px"], s["median_nm"]))
    print("  {:<18}: {:>8.2f} px   {:>10.2f} nm".format("Min", s["min_px"], s["min_nm"]))
    print("  {:<18}: {:>8.2f} px   {:>10.2f} nm".format("Max", s["max_px"], s["max_nm"]))

    print(line)
    print("")


# ============================================================
# STEP 7 -- Per-image driver: load, detect (silent), analyse, print, plot
# ============================================================

def analyze_image(img_path, roi_label):
    img_gray = load_image(img_path)

    tophat_norm, tophat_smooth = preprocess(img_gray)
    candidates_df = detect_blobs(tophat_smooth)
    validated_df, final_mask, extracted = validate_precipitates(img_gray, tophat_norm, candidates_df)

    if len(validated_df) == 0:
        raise ValueError(f"No precipitates validated for {roi_label} -- check detection parameters / image path.")

    mpp = SCALE_BAR_MICRONS / SCALE_BAR_PIXELS
    npp = mpp * 1000

    d_stats = diameter_stats(validated_df, mpp)
    nn_spacing, nn_index, s_stats = spacing_stats(validated_df, mpp, EDGE_TO_EDGE)

    result_df = validated_df[["centroid_x", "centroid_y", "equivalent_diameter_px"]].copy().reset_index(drop=True)
    result_df["equivalent_diameter_um"] = result_df["equivalent_diameter_px"] * mpp
    result_df["equivalent_diameter_nm"] = result_df["equivalent_diameter_px"] * npp
    result_df["nn_spacing_px"] = nn_spacing
    result_df["nn_spacing_um"] = nn_spacing * mpp
    result_df["nn_spacing_nm"] = nn_spacing * npp
    result_df["nn_index"] = nn_index
    result_df["roi"] = roi_label

    print_summary(d_stats, s_stats, mpp, npp, roi_label)
    plot_summary(img_gray, validated_df, result_df, d_stats, s_stats, roi_label)

    return {
        "roi": roi_label,
        "diameter": d_stats,
        "spacing": s_stats,
        "result_df": result_df,
        "n": d_stats["n"],
    }


# ============================================================
# STEP 8 -- Pool the 3 sub-images (PRINT ONLY, no plots)
# ============================================================

def pooled_stats_nm(values):
    """Descriptive statistics computed directly from pooled raw nm values."""
    values = np.asarray(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    n = len(values)

    return {
        "mean_nm": mean,
        "std_nm": std,
        "median_nm": float(np.median(values)),
        "min_nm": float(np.min(values)),
        "max_nm": float(np.max(values)),
        "cv": std / mean,
        "sem_nm": std / np.sqrt(n),
        "ci95_nm": 1.96 * std / np.sqrt(n),
        "n": n,
    }


def pool_results(results_list):
    dfs = [res["result_df"] for res in results_list]
    combined_df = pd.concat(dfs, ignore_index=True)

    per_roi = []
    for res in results_list:
        per_roi.append({
            "label": res["roi"],
            "n": res["n"],
            "spacing_mean_nm": res["spacing"]["mean_nm"],
            "spacing_std_nm": res["spacing"]["std_nm"],
            "spacing_median_nm": res["spacing"]["median_nm"],
            "diam_mean_nm": res["diameter"]["mean_nm"],
            "diam_std_nm": res["diameter"]["std_nm"],
            "diam_median_nm": res["diameter"]["median_nm"],
        })

    pooled_spacing = pooled_stats_nm(combined_df["nn_spacing_nm"].values)
    pooled_diameter = pooled_stats_nm(combined_df["equivalent_diameter_nm"].values)

    return {
        "spacing": pooled_spacing,
        "diameter": pooled_diameter,
        "result_df": combined_df,
        "per_roi": per_roi,
        "n_total": len(combined_df),
        "n_rois": len(results_list),
    }


def print_pooled_summary(pooled):
    sp = pooled["spacing"]
    di = pooled["diameter"]
    line = "=" * 70

    print("")
    print(line)
    print("  POOLED PRECIPITATE ANALYSIS  --  725 C, 5 hr")
    print("  {} ROIs  |  {} particles total".format(pooled["n_rois"], pooled["n_total"]))
    print(line)

    print("")
    print("  PER-ROI BREAKDOWN")
    print("  {:<10} {:>6}  {:>16}  {:>14}  {:>12}  {:>11}".format(
        "ROI", "n", "Spacing mean", "Spacing std", "Diam mean", "Diam std"))
    print("  {} {}  {}  {}  {}  {}".format(
        "-" * 10, "-" * 6, "-" * 16, "-" * 14, "-" * 12, "-" * 11))

    for r in pooled["per_roi"]:
        print("  {:<10} {:>6}  {:>13.2f} nm  {:>11.2f} nm  {:>9.2f} nm  {:>8.2f} nm".format(
            r["label"], r["n"], r["spacing_mean_nm"], r["spacing_std_nm"],
            r["diam_mean_nm"], r["diam_std_nm"]))

    print("")
    print("  POOLED NEAREST-NEIGHBOUR SPACING, s   (n = {})".format(sp["n"]))
    print("  {:<18}: {:>10.2f} nm".format("Mean", sp["mean_nm"]))
    print("  {:<18}: {:>10.2f} nm".format("Std Dev", sp["std_nm"]))
    print("  {:<18}: {:.2f} +/- {:.2f} nm".format("95% CI", sp["mean_nm"], sp["ci95_nm"]))
    print("  {:<18}: {:>10.2f} nm".format("SEM", sp["sem_nm"]))
    print("  {:<18}: {:>10.2f} nm".format("Median", sp["median_nm"]))
    print("  {:<18}: {:>10.2f} nm".format("Min", sp["min_nm"]))
    print("  {:<18}: {:>10.2f} nm".format("Max", sp["max_nm"]))
    print("  {:<18}: {:>10.3f}".format("CV", sp["cv"]))

    print("")
    print("  POOLED PRECIPITATE DIAMETER, d   (n = {})".format(di["n"]))
    print("  {:<18}: {:>10.2f} nm".format("Mean", di["mean_nm"]))
    print("  {:<18}: {:>10.2f} nm".format("Std Dev", di["std_nm"]))
    print("  {:<18}: {:.2f} +/- {:.2f} nm".format("95% CI", di["mean_nm"], di["ci95_nm"]))
    print("  {:<18}: {:>10.2f} nm".format("SEM", di["sem_nm"]))
    print("  {:<18}: {:>10.2f} nm".format("Median", di["median_nm"]))
    print("  {:<18}: {:>10.2f} nm".format("Min", di["min_nm"]))
    print("  {:<18}: {:>10.2f} nm".format("Max", di["max_nm"]))
    print("  {:<18}: {:>10.3f}".format("CV", di["cv"]))

    print(line)
    print("")

    print("  REPORTABLE VALUES (mean +/- 1 SD):")
    print("  Interparticle spacing : {:.2f} +/- {:.2f} nm  (n = {})".format(
        sp["mean_nm"], sp["std_nm"], sp["n"]))
    print("  Precipitate diameter  : {:.2f} +/- {:.2f} nm  (n = {})".format(
        di["mean_nm"], di["std_nm"], di["n"]))
    print("")


# ============================================================
# MAIN -- run each image (print + plot), then pool (print only)
# ============================================================

if __name__ == "__main__":

    results_list = []
    for roi_label, img_path in IMAGE_PATHS.items():
        res = analyze_image(img_path, roi_label)
        results_list.append(res)

    pooled = pool_results(results_list)
    print_pooled_summary(pooled)
