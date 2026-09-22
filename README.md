# LUNAR-X (ISRO SIH 2026 - Problem Statement SIH26166)

> **Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC-2 and IIRS)**

**Organization:** Indian Space Research Organisation (ISRO)  
**Domain:** Space Technology  
**Project Name:** LUNAR-X  

---

## 🛰️ Project Overview

**LUNAR-X** is an AI/Computer-Vision and Remote Sensing system designed for ISRO's Chandrayaan-2 lunar optical and hyperspectral payload datasets (OHRC, TMC-2, and IIRS). The system computes robust, spatially distributed image correspondences, geometric transformations, and high-accuracy registration between lunar imagery acquired under different spatial resolutions, viewing angles, sun illumination conditions, rotations, and sensor modalities.

---

## 💻 Web Dashboard & SIH Demonstration Application

LUNAR-X features a space-themed interactive Streamlit Web Application for live SIH 2026 demonstrations:

```bash
streamlit run app.py
```

See [`DEMO_GUIDE.md`](file:///c:/Users/laksh/OneDrive/Desktop/LUNAR-X/DEMO_GUIDE.md) for the 3–5 minute step-by-step judge presentation walkthrough.

---

## 🏗️ System Architecture & Complete Modules

LUNAR-X integrates OHRC optical imagery (~0.25 m/px), TMC-2 terrain mapping imagery (~5.0 m/px), and IIRS hyperspectral imagery (~100.0 m/px across 256 bands) into a unified tri-sensor correspondence framework:

```
LUNAR-X/
├── app.py                              # Master Streamlit Web Application UI
├── config.yaml                         # Central YAML configuration
├── requirements.txt                    # Dependencies (PyTorch, OpenCV, NumPy, SciPy, Streamlit, PyYAML)
├── README.md                           # Master Project Documentation & execution guide
├── DEMO_GUIDE.md                       # SIH 2026 3-5 Minute Judge Presentation Script
├── backend/
│   ├── core/                           # Standardized LunarImage & DatasetManifest
│   ├── sensors/
│   │   ├── ohrc/                       # OHRC Validator, Reader, & Real Pipeline (Modules 1, 2, 7)
│   │   ├── tmc2/                       # TMC-2 Validator, Reader, & Preprocessor (Module 8)
│   │   └── iirs/                       # IIRS Validator, Reader, Preprocessor, & PCA (Modules 11, 12)
│   ├── preprocessing/                  # Preprocessing Core & Image Tiling (Modules 3, 4)
│   ├── correspondence/                 # SIFT, ORB, AKAZE Feature Baselines (Module 5)
│   ├── matching/                       # Multi-Scale & Cross-Modal Matchers (Modules 9, 13, 14)
│   │   ├── cross_sensor.py             # OHRC ↔ TMC-2 scale-resampled matcher
│   │   ├── cross_modal.py              # Optical (OHRC/TMC-2) ↔ IIRS gradient-magnitude matcher
│   │   └── tri_sensor.py               # Unified Tri-Sensor Consensus Engine
│   ├── evaluation/                     # Evaluation Framework (Modules 6, 10)
│   └── visualization/                  # Visual Panel Generators (Module 10)
├── data/                               # Dataset & Manifest management
├── outputs/                            # Visual panels & benchmark JSON/CSV reports
├── scripts/
│   ├── run_full_system.py              # Complete System End-to-End Master Runner
│   └── ...
└── tests/                              # PyTest test suite (34/34 passing)
```

---

## ⚡ Quick Start & Commands

### 1. Launch Interactive Web App
```bash
streamlit run app.py
```

### 2. Run Complete Test Suite (34 / 34 Tests Passing)
```bash
pytest tests/ -v
```

### 3. Run Master End-to-End CLI Pipeline
```bash
python scripts/run_full_system.py --config config.yaml
```
