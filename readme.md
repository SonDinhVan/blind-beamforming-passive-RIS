# Near-optimal Reconfigurable Intelligent Surface Configuration: Blind Beamforming with Sensing (BORN)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Conda env](https://img.shields.io/badge/conda-environment.yml-brightgreen.svg)
![License](https://img.shields.io/badge/License-All%20rights%20reserved-lightgrey.svg)

</div>

---

## 🧭 Overview

This repository accompanies the paper:

**Son Dinh-Van, Nam Phuong Tran, Matthew D. Higgins, “Near-optimal Reconfigurable Intelligent Surface Configuration: Blind Beamforming with Sensing.”**

It provides:

- An implementation of **BORN**, a blind beamforming algorithm that uses only RSS measurements  
- Notebooks for analysis and simulation  
- Reproducible simulations for figures in the paper

---

## ⚙️ Installation

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

A reproducible environment via Conda is recommended:

```bash
conda env create -f environment.yml
conda activate <env-name>
```

---

## 🚀 Quick start

Open the analysis notebook to run **BORN** end-to-end (data collection simulation → sensing → optimization → evaluation):

```
notebooks/analysis.ipynb
```

---

## 📊 Reproducing results

Run the notebooks below to reproduce the simulation figures.  
Each notebook is organized for top-to-bottom execution with annotated steps and figure generation:

```
notebooks/simulation results/
```

---

## 📬 Contact

Questions and feedback are welcome.  
**Email:** son.v.dinh@warwick.ac.uk

---

## © Copyright

© 2025 **Son Dinh-Van**. All rights reserved.
