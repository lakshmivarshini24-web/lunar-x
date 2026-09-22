"""
LUNAR-X SIH 2026 Master Demonstration Application

ISRO Chandrayaan-2 Multi-Modal Lunar Image Correspondence System
Problem Statement ID: SIH26166

Interactive 8-Page Space-Themed Dashboard integrating Phase 1, Phase 2, and Phase 3 modules:
- Page 1: Home (Mission Overview & SIH Context)
- Page 2: Data Input & Validation (Demo Mode vs Real ISRO Data Mode)
- Page 3: Preprocessing & PCA Dimensionality Reduction
- Page 4: Pairwise Correspondence Analysis (OHRC<->TMC2, OHRC<->IIRS, TMC2<->IIRS)
- Page 5: Tri-Sensor Consensus View (OHRC <-> TMC-2 <-> IIRS Loop Closure)
- Page 6: Quantitative Results Dashboard (Empirical Metrics)
- Page 7: Benchmark Comparison (Phase 2 Classical vs Phase 3 Improved Cross-Modal)
- Page 8: Export Analysis Report (JSON/CSV Downloads)
"""

import json
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import yaml

from backend.core.lunar_image import LunarImage
from backend.core.manifest import DatasetManifest
from backend.sensors.ohrc.validator import OHRCValidator
from backend.sensors.ohrc.reader import load_ohrc
from backend.sensors.ohrc.pipeline import OHRCRealCorrespondencePipeline
from backend.sensors.tmc2.validator import TMC2Validator
from backend.sensors.tmc2.reader import load_tmc2
from backend.sensors.tmc2.preprocessor import TMC2Preprocessor
from backend.sensors.iirs.validator import IIRSValidator
from backend.sensors.iirs.reader import load_iirs
from backend.sensors.iirs.preprocessor import IIRSPreprocessor
from backend.matching.cross_sensor import CrossSensorMatcher
from backend.matching.cross_modal import CrossModalMatcher
from backend.matching.tri_sensor import TriSensorCorrespondenceEngine
from backend.evaluation.metrics import compute_metrics
from backend.evaluation.cross_sensor_eval import CrossSensorEvaluator
from backend.evaluation.tri_sensor_eval import TriSensorEvaluator
from backend.visualization.cross_sensor_vis import visualize_cross_sensor_correspondence
from backend.visualization.tri_sensor_vis import visualize_tri_sensor_correspondence

# Streamlit Page Setup
st.set_page_config(
    page_title="LUNAR-X | ISRO SIH 2026",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Space Theme CSS
st.html(
    """
    <style>
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0f172a 100%);
    }
    .css-1d37w0e, .stSidebar {
        background-color: #0d1322 !important;
        border-right: 1px solid #1e293b;
    }
    h1, h2, h3 {
        color: #38bdf8 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background: linear-gradient(90deg, #0284c7 0%, #2563eb 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0369a1 0%, #1d4ed8 100%);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6);
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .tag-real {
        background-color: #059669;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .tag-demo {
        background-color: #d97706;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    </style>
    """
)


@st.cache_data
def get_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = get_config()

    st.sidebar.image("https://img.icons8.com/isometric-line/100/moon.png", width=64)
    st.sidebar.title("LUNAR-X Navigation")
    st.sidebar.caption("ISRO SIH 2026 | PS ID: SIH26166")

    page = st.sidebar.radio(
        "Go to Page:",
        [
            "🏠 1. Home",
            "📥 2. Data Input & Validation",
            "⚙️ 3. Preprocessing & PCA",
            "🔗 4. Pairwise Correspondence",
            "🔺 5. Tri-Sensor Consensus View",
            "📊 6. Quantitative Results",
            "⚔️ 7. Benchmark Comparison",
            "📄 8. Export Analysis Report",
        ],
    )

    # Global Session State Initializations
    if "data_mode" not in st.session_state:
        st.session_state.data_mode = "Demo / Synthetic Data Mode"
    if "pipeline_run_completed" not in st.session_state:
        st.session_state.pipeline_run_completed = False

    # -------------------------------------------------------------------------
    # PAGE 1: HOME
    # -------------------------------------------------------------------------
    if page == "🏠 1. Home":
        st.title("🌙 LUNAR-X: Multi-Modal Lunar Image Correspondence")
        st.subheader("Indian Space Research Organisation (ISRO) — SIH 2026 Project")
        st.markdown("**Problem Statement SIH26166:** *Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS).*")

        st.info(
            "💡 **Why LUNAR-X is Essential:** Different lunar sensors capture the same terrain with different spatial resolutions (~0.25m OHRC vs ~5m TMC-2 vs ~100m IIRS), spectral responses (optical reflectance vs infrared emission), illumination conditions, and viewing geometries. **LUNAR-X** computes robust, spatially invariant correspondences and geometric registration across all modalities."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
                ### 🛰️ OHRC
                **Orbiter High Resolution Camera**
                - Resolution: **0.25 m/pixel**
                - High-detail crater geometry
                - Optical panchromatic band
                """
            )
        with col2:
            st.markdown(
                """
                ### 🗺️ TMC-2
                **Terrain Mapping Camera-2**
                - Resolution: **5.0 m/pixel**
                - Stereo & orthorectified context
                - Surface elevation mapping
                """
            )
        with col3:
            st.markdown(
                """
                ### 🌈 IIRS
                **Imaging Infrared Spectrometer**
                - Resolution: **100.0 m/pixel**
                - Hyperspectral (256 bands)
                - Mineral absorption & IR spectra
                """
            )

        st.markdown("---")
        if st.button("🚀 Start Correspondence Analysis"):
            st.session_state.current_page = "📥 2. Data Input & Validation"
            st.rerun()

    # -------------------------------------------------------------------------
    # PAGE 2: DATA INPUT & VALIDATION
    # -------------------------------------------------------------------------
    elif page == "📥 2. Data Input & Validation":
        st.title("📥 Dataset Input & Multi-Sensor Validation")

        st.session_state.data_mode = st.radio(
            "Select Processing Mode:",
            ["Demo / Synthetic Data Mode", "Real Chandrayaan-2 Data Mode"],
            horizontal=True,
        )

        if st.session_state.data_mode == "Demo / Synthetic Data Mode":
            st.html('<span class="tag-demo">DEMO / SYNTHETIC DATA MODE ACTIVE</span>')
            st.caption("Using controlled synthetic Chandrayaan-2 test triplets with ground-truth homography for demonstration.")

            ohrc_p = "data/raw/synthetic_trisensor/ohrc_tri_01.tif"
            tmc2_p = "data/raw/synthetic_trisensor/tmc2_tri_01.tif"
            iirs_p = "data/raw/synthetic_trisensor/iirs_tri_01.npy"

            # Auto-generate if missing
            if not Path(ohrc_p).exists():
                from scripts.generate_trisensor_data import generate_trisensor_triplet
                generate_trisensor_triplet(Path("data/raw/synthetic_trisensor"), triplet_id=1)

            st.session_state.ohrc_path = ohrc_p
            st.session_state.tmc2_path = tmc2_p
            st.session_state.iirs_path = iirs_p

        else:
            st.html('<span class="tag-real">REAL CHANDRAYAAN-2 DATA MODE ACTIVE</span>')
            st.caption("Select local ISRO PDS4 / ISDA datasets from configurable directories or upload custom payload files.")

            real_paths_cfg = cfg.get("paths", {})
            real_ohrc_dir = Path(real_paths_cfg.get("real_ohrc_dir", "data/real/ohrc"))
            real_tmc2_dir = Path(real_paths_cfg.get("real_tmc2_dir", "data/real/tmc2"))
            real_iirs_dir = Path(real_paths_cfg.get("real_iirs_dir", "data/real/iirs"))

            for d in [real_ohrc_dir, real_tmc2_dir, real_iirs_dir]:
                d.mkdir(parents=True, exist_ok=True)

            def get_files(directory: Path, exts: tuple) -> list:
                if not directory.exists():
                    return []
                return sorted([f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in exts])

            ohrc_files = get_files(real_ohrc_dir, (".tif", ".tiff", ".png", ".jpg", ".jpeg"))
            tmc2_files = get_files(real_tmc2_dir, (".tif", ".tiff", ".png", ".jpg", ".jpeg"))
            iirs_files = get_files(real_iirs_dir, (".npy", ".tif", ".tiff", ".png"))

            col_sel1, col_sel2, col_sel3 = st.columns(3)

            # --- OHRC Selection ---
            with col_sel1:
                st.subheader("📷 OHRC Dataset")
                ohrc_options = [f.name for f in ohrc_files] + ["Upload Custom File..."]
                ohrc_choice = st.selectbox("Select Local OHRC File:", ohrc_options, index=0 if ohrc_files else len(ohrc_options)-1, key="sel_ohrc")

                if ohrc_choice == "Upload Custom File...":
                    up_ohrc = st.file_uploader("Upload OHRC Raster (.tif, .png)", type=["tif", "tiff", "png", "jpg"], key="up_ohrc")
                    if up_ohrc:
                        tmp_dir = Path("data/raw/user_uploaded")
                        tmp_dir.mkdir(parents=True, exist_ok=True)
                        p_o = tmp_dir / up_ohrc.name
                        p_o.write_bytes(up_ohrc.getbuffer())
                        st.session_state.ohrc_path = str(p_o)
                elif ohrc_files:
                    selected_file = [f for f in ohrc_files if f.name == ohrc_choice][0]
                    st.session_state.ohrc_path = str(selected_file)

                selected_ohrc_name = Path(st.session_state.ohrc_path).name if "ohrc_path" in st.session_state else "None"
                st.markdown(f"**Selected File:** `{selected_ohrc_name}`")
                st.caption(f"Path: `{st.session_state.get('ohrc_path', 'None')}`")

            # --- TMC-2 Selection ---
            with col_sel2:
                st.subheader("🏔️ TMC-2 Dataset")
                tmc2_options = [f.name for f in tmc2_files] + ["Upload Custom File..."]
                tmc2_choice = st.selectbox("Select Local TMC-2 File:", tmc2_options, index=0 if tmc2_files else len(tmc2_options)-1, key="sel_tmc2")

                if tmc2_choice == "Upload Custom File...":
                    up_tmc2 = st.file_uploader("Upload TMC-2 Raster (.tif, .png)", type=["tif", "tiff", "png", "jpg"], key="up_tmc2")
                    if up_tmc2:
                        tmp_dir = Path("data/raw/user_uploaded")
                        tmp_dir.mkdir(parents=True, exist_ok=True)
                        p_t = tmp_dir / up_tmc2.name
                        p_t.write_bytes(up_tmc2.getbuffer())
                        st.session_state.tmc2_path = str(p_t)
                elif tmc2_files:
                    selected_file = [f for f in tmc2_files if f.name == tmc2_choice][0]
                    st.session_state.tmc2_path = str(selected_file)

                selected_tmc2_name = Path(st.session_state.tmc2_path).name if "tmc2_path" in st.session_state else "None"
                st.markdown(f"**Selected File:** `{selected_tmc2_name}`")
                st.caption(f"Path: `{st.session_state.get('tmc2_path', 'None')}`")

            # --- IIRS Selection ---
            with col_sel3:
                st.subheader("🌈 IIRS Hyperspectral")
                iirs_options = [f.name for f in iirs_files] + ["Upload Custom File..."]
                iirs_choice = st.selectbox("Select Local IIRS File:", iirs_options, index=0 if iirs_files else len(iirs_options)-1, key="sel_iirs")

                if iirs_choice == "Upload Custom File...":
                    up_iirs = st.file_uploader("Upload IIRS Array (.npy, .tif)", type=["npy", "tif", "tiff"], key="up_iirs")
                    if up_iirs:
                        tmp_dir = Path("data/raw/user_uploaded")
                        tmp_dir.mkdir(parents=True, exist_ok=True)
                        p_i = tmp_dir / up_iirs.name
                        p_i.write_bytes(up_iirs.getbuffer())
                        st.session_state.iirs_path = str(p_i)
                elif iirs_files:
                    selected_file = [f for f in iirs_files if f.name == iirs_choice][0]
                    st.session_state.iirs_path = str(selected_file)

                selected_iirs_name = Path(st.session_state.iirs_path).name if "iirs_path" in st.session_state else "None"
                st.markdown(f"**Selected File:** `{selected_iirs_name}`")
                st.caption(f"Path: `{st.session_state.get('iirs_path', 'None')}`")

        st.markdown("### Sensor Dataset Validation Suite")
        if st.button("🔍 Validate Datasets"):
            v_o = OHRCValidator(cfg).validate_file(st.session_state.ohrc_path)
            v_t = TMC2Validator(cfg).validate_file(st.session_state.tmc2_path)
            v_i = IIRSValidator(cfg).validate_file(st.session_state.iirs_path)

            col1, col2, col3 = st.columns(3)
            with col1:
                if v_o["is_valid"]:
                    st.success("OHRC Valid")
                else:
                    st.error("OHRC Invalid")
                st.json(v_o)
            with col2:
                if v_t["is_valid"]:
                    st.success("TMC-2 Valid")
                else:
                    st.error("TMC-2 Invalid")
                st.json(v_t)
            with col3:
                if v_i["is_valid"]:
                    st.success("IIRS Valid")
                else:
                    st.error("IIRS Invalid")
                st.json(v_i)

        st.markdown("---")
        if st.button("⚡ RUN FULL LUNAR-X PIPELINE"):
            with st.spinner("Executing Multi-Modal Preprocessing, PCA Reduction, and Cross-Modal Tri-Sensor Matching..."):
                progress_bar = st.progress(0)

                # 1. Ingest
                l_ohrc = load_ohrc(st.session_state.ohrc_path, cfg)
                l_tmc2 = load_tmc2(st.session_state.tmc2_path, cfg)
                l_iirs_raw = load_iirs(st.session_state.iirs_path, cfg)
                progress_bar.progress(25)

                # 2. Preprocess & PCA
                prep_tmc2 = TMC2Preprocessor(cfg).process(l_tmc2)
                l_iirs_pc1, iirs_diag = IIRSPreprocessor(cfg).process(l_iirs_raw)
                progress_bar.progress(50)

                # 3. Tri-Sensor Engine
                tri_engine = TriSensorCorrespondenceEngine(cfg)
                tri_res = tri_engine.process_triplet(l_ohrc, prep_tmc2, l_iirs_pc1)
                progress_bar.progress(80)

                # 4. Evaluation & Reports
                evaluator = TriSensorEvaluator(cfg)
                dataset_type = "REAL_ISRO_PDS4" if "REAL" in st.session_state.data_mode else "SYNTHETIC_TRISENSOR_TRIPLET"
                metrics_summary = evaluator.evaluate(tri_res, dataset_type=dataset_type)
                evaluator.save_report(metrics_summary, output_dir="outputs/benchmarks")

                # Store in session
                st.session_state.l_ohrc = l_ohrc
                st.session_state.l_tmc2 = prep_tmc2
                st.session_state.l_iirs_pc1 = l_iirs_pc1
                st.session_state.iirs_diag = iirs_diag
                st.session_state.tri_res = tri_res
                st.session_state.metrics_summary = metrics_summary
                st.session_state.pipeline_run_completed = True

                progress_bar.progress(100)
                st.success("🎉 LUNAR-X Pipeline Execution Completed Successfully!")

    # -------------------------------------------------------------------------
    # PAGE 3: PREPROCESSING & PCA
    # -------------------------------------------------------------------------
    elif page == "⚙️ 3. Preprocessing & PCA":
        st.title("⚙️ Image Preprocessing & Hyperspectral PCA Reduction")

        if not st.session_state.pipeline_run_completed:
            st.warning("Please run the pipeline from Page 2 first!")
            return

        st.subheader("IIRS Hyperspectral SVD PCA Dimensionality Reduction")
        diag = st.session_state.iirs_diag
        col1, col2, col3 = st.columns(3)
        col1.metric("Original Spectral Bands", diag["num_original_bands"])
        col2.metric("Filtered Bad Bands", diag["num_bad_bands"])
        col3.metric("Cumulative PCA Variance", f"{diag['cumulative_explained_variance']*100:.2f}%")

        st.markdown("#### Raw vs Preprocessed Visual Inspection")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.image(st.session_state.l_ohrc.to_uint8(), caption="OHRC Preprocessed (0.25 m/px)", use_container_width=True)
        with col_b:
            st.image(st.session_state.l_tmc2.to_uint8(), caption="TMC-2 Preprocessed (5.0 m/px)", use_container_width=True)
        with col_c:
            st.image(st.session_state.l_iirs_pc1.to_uint8(), caption="IIRS PC1 Component (100.0 m/px)", use_container_width=True)

    # -------------------------------------------------------------------------
    # PAGE 4: PAIRWISE CORRESPONDENCE ANALYSIS
    # -------------------------------------------------------------------------
    elif page == "🔗 4. Pairwise Correspondence":
        st.title("🔗 Pairwise Feature Correspondence Analysis")

        if not st.session_state.pipeline_run_completed:
            st.warning("Please run the pipeline from Page 2 first!")
            return

        res = st.session_state.tri_res
        tab1, tab2, tab3 = st.tabs(["OHRC ↔ TMC-2", "OHRC ↔ IIRS", "TMC-2 ↔ IIRS"])

        with tab1:
            r = res.res_ohrc_tmc2
            st.markdown(f"### OHRC (0.25m) ↔ TMC-2 (5.0m) Across 20x Resolution Scale Jump")
            st.write(f"**RANSAC Inliers:** {r.num_inliers} / {r.num_filtered_matches} ({r.inlier_ratio*100:.1f}%) | **Confidence:** {r.confidence_score:.3f}")
            fig, ax = plt.subplots(figsize=(10, 5))
            canvas = draw_match_canvas_ui(st.session_state.l_ohrc.to_uint8(), st.session_state.l_tmc2.to_uint8(), r.inlier_pts_a_native, r.inlier_pts_b_native)
            ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            st.pyplot(fig)

        with tab2:
            r = res.res_ohrc_iirs
            st.markdown(f"### OHRC (0.25m) ↔ IIRS (100.0m) Cross-Modal Correspondence")
            st.write(f"**RANSAC Inliers:** {r.num_inliers} / {r.num_filtered_matches} ({r.inlier_ratio*100:.1f}%) | **Confidence:** {r.confidence_score:.3f}")
            fig, ax = plt.subplots(figsize=(10, 5))
            canvas = draw_match_canvas_ui(st.session_state.l_ohrc.to_uint8(), st.session_state.l_iirs_pc1.to_uint8(), r.inlier_pts_a_native, r.inlier_pts_b_native)
            ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            st.pyplot(fig)

        with tab3:
            r = res.res_tmc2_iirs
            st.markdown(f"### TMC-2 (5.0m) ↔ IIRS (100.0m) Cross-Modal Correspondence")
            st.write(f"**RANSAC Inliers:** {r.num_inliers} / {r.num_filtered_matches} ({r.inlier_ratio*100:.1f}%) | **Confidence:** {r.confidence_score:.3f}")
            fig, ax = plt.subplots(figsize=(10, 5))
            canvas = draw_match_canvas_ui(st.session_state.l_tmc2.to_uint8(), st.session_state.l_iirs_pc1.to_uint8(), r.inlier_pts_a_native, r.inlier_pts_b_native)
            ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            st.pyplot(fig)

    # -------------------------------------------------------------------------
    # PAGE 5: TRI-SENSOR CONSENSUS VIEW
    # -------------------------------------------------------------------------
    elif page == "🔺 5. Tri-Sensor Consensus View":
        st.title("🔺 Unified Tri-Sensor Correspondence Consensus")

        if not st.session_state.pipeline_run_completed:
            st.warning("Please run the pipeline from Page 2 first!")
            return

        res = st.session_state.tri_res
        st.markdown(
            """
            ```
                      IIRS (100.0m Hyperspectral)
                     /                           \
                    /                             \
             OHRC (0.25m Optical) <---------> TMC-2 (5.0m Stereo)
            ```
            """
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Tri-Sensor Agreement Score", f"{res.tri_sensor_agreement_score:.4f}")
        col2.metric("Loop Closure Error", f"{res.loop_closure_error_px:.2f} px")
        col3.metric("Consensus Status", "PASSED" if res.is_consensus_valid else "FAILED")

    # -------------------------------------------------------------------------
    # PAGE 6: QUANTITATIVE RESULTS
    # -------------------------------------------------------------------------
    elif page == "📊 6. Quantitative Results":
        st.title("📊 Empirical Calculated Metrics")

        if not st.session_state.pipeline_run_completed:
            st.warning("Please run the pipeline from Page 2 first!")
            return

        m = st.session_state.metrics_summary
        st.json(m)

    # -------------------------------------------------------------------------
    # PAGE 7: BENCHMARK COMPARISON
    # -------------------------------------------------------------------------
    elif page == "⚔️ 7. Benchmark Comparison":
        st.title("⚔️ Phase 2 Classical Baseline vs. Phase 3 Improved Pipeline")

        st.markdown(
            """
            | Robustness Dimension | Phase 2 Classical Baseline | Phase 3 Improved Cross-Modal Pipeline |
            | :--- | :--- | :--- |
            | **Scale Invariance** | Limited to ~2× scale ratio | **Supports ~20× to 400× Resolution Scale Pyramids** |
            | **Cross-Modal Invariance** | Fails on optical vs IR contrast inversion | **Sobel Gradient Magnitude Representation + CLAHE** |
            | **Hyperspectral Data** | Single-band grayscale only | **SVD-based PCA (99.00% Variance Preservation)** |
            | **Multi-Sensor Alignment** | Pairwise matching only | **Unified Tri-Sensor Geometric Loop Closure Consensus** |
            """
        )

    # -------------------------------------------------------------------------
    # PAGE 8: EXPORT REPORT
    # -------------------------------------------------------------------------
    elif page == "📄 8. Export Analysis Report":
        st.title("📄 Export LUNAR-X Analysis Report")

        if not st.session_state.pipeline_run_completed:
            st.warning("Please run the pipeline from Page 2 first!")
            return

        m_json = json.dumps(st.session_state.metrics_summary, indent=2)
        st.download_button(
            label="💾 Download JSON Report",
            data=m_json,
            file_name="lunar_x_analysis_report.json",
            mime="application/json",
        )


def draw_match_canvas_ui(img1, img2, pts1, pts2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas_h = max(h1, h2)
    canvas_w = w1 + w2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    canvas[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1
    canvas[:h2, w1 : w1 + w2] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if img2.ndim == 2 else img2

    if len(pts1) > 0 and len(pts2) > 0:
        for (x1, y1), (x2, y2) in zip(pts1, pts2):
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2 + w1), int(y2))
            cv2.line(canvas, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(canvas, pt1, 3, (255, 0, 0), -1)
            cv2.circle(canvas, pt2, 3, (0, 0, 255), -1)

    return canvas


if __name__ == "__main__":
    main()
