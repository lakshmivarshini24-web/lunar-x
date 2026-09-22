# LUNAR-X — SIH 2026 Demonstration Guide (3–5 Minute Judge Presentation)

**Organization:** Indian Space Research Organisation (ISRO)  
**Problem Statement ID:** SIH26166  
**Title:** Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)  
**Application Entrypoint:** `streamlit run app.py`  

---

## 🎯 1-Minute Pitch for SIH Judges

> *"Respected Judges, Chandrayaan-2 features three distinct sensors—OHRC at 0.25m resolution, TMC-2 at 5m resolution, and IIRS hyperspectral imagery at 100m resolution across 256 spectral bands. Standard computer vision algorithms fail when attempting to match images across a 20x to 400x resolution jump, different illumination angles, and optical vs infrared contrast inversions. **LUNAR-X** is an AI/Remote Sensing system that computes robust, scale-invariant, and modality-invariant correspondences across all three Chandrayaan-2 payloads."*

---

## 🧭 Step-by-Step 3–5 Minute Demo Walkthrough

### Step 1: Open Application & Project Introduction (Page 1 — Home)
1. Run `streamlit run app.py` in terminal.
2. Direct the judges to **Page 1 (Home)**.
3. Highlight the 3 instrument cards:
   - **OHRC** (0.25 m/px optical high-res)
   - **TMC-2** (5.0 m/px stereo context)
   - **IIRS** (100.0 m/px hyperspectral 256 bands)
4. Click **"🚀 Start Correspondence Analysis"**.

### Step 2: Select Dataset & Execute Pipeline (Page 2 — Data Input)
1. Select **"Demo / Synthetic Data Mode"** (or upload custom rasters in **Real Data Mode**).
2. Point out the clear visual tag: `<span class="tag-demo">DEMO / SYNTHETIC DATA MODE ACTIVE</span>` (ensuring transparency).
3. Click **"🔍 Validate Datasets"** to show instant dimension, bit-depth, and format integrity checks.
4. Click **"⚡ RUN FULL LUNAR-X PIPELINE"**.
5. Show the live progress bar performing:
   - Data Ingestion
   - Hyperspectral SVD PCA Reduction
   - Multi-Scale Gradient-Magnitude Feature Extraction
   - Pairwise RANSAC Homography Estimation
   - Tri-Sensor Loop Closure Consensus

### Step 3: Inspect Preprocessing & PCA Reduction (Page 3 — Preprocessing & PCA)
1. Navigate to **Page 3**.
2. Show the IIRS SVD PCA Reduction metrics:
   - **32 Original Spectral Bands** reduced to **3 Principal Components**
   - **Cumulative Explained Variance: 99.00%**
3. Show raw vs preprocessed raster displays (CLAHE contrast stretch & illumination normalization).

### Step 4: Explore Pairwise Cross-Modal Matches (Page 4 — Pairwise Correspondence)
1. Navigate to **Page 4**.
2. Click through the 3 tabs:
   - **OHRC ↔ TMC-2**: Demonstrate scale-resampled feature matching across a 20× spatial resolution jump.
   - **OHRC ↔ IIRS**: Demonstrate gradient-domain cross-modal matching (optical reflectance vs infrared emission).
   - **TMC-2 ↔ IIRS**: Demonstrate context-level cross-modal matching.

### Step 5: Demonstrate Tri-Sensor Consensus (Page 5 — Tri-Sensor View)
1. Navigate to **Page 5**.
2. Show the Tri-Sensor loop closure topology:
   ```
             IIRS (100.0m)
            /             \
           /               \
     OHRC (0.25m) <---------> TMC-2 (5.0m)
   ```
3. Highlight the calculated **Tri-Sensor Agreement Score** and **Loop Closure Error (px)**.

### Step 6: Empirical Results & Report Export (Pages 6, 7 & 8)
1. Show **Page 6 (Quantitative Results)** for raw calculated numbers (Keypoints, Inliers, Inlier Ratio, Reprojection Error, FPS).
2. Show **Page 7 (Benchmark Comparison)** comparing Phase 2 Classical Baseline against Phase 3 Improved Pipeline.
3. On **Page 8 (Export Report)**, click **"💾 Download JSON Report"** to show machine-readable report export.

---

## ⚡ Quick Start Command

To launch the web dashboard locally:

```bash
streamlit run app.py
```
