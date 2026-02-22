# 🏗️ Professional Space Truss Suite (3D)

**A Pedagogical & Commercial-Grade Structural Analysis Tool**

Developed by **Mr. D Mandal**, Assistant Professor, Department of Civil Engineering, KITS Ramtek.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-GPLv3-green.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

## 📖 Software Overview

The **Professional Space Truss Suite** is an interactive, web-based environment for the linear static analysis of 3D spatial structures using the **Direct Stiffness Method (DSM)**. 

Unlike commercial "black-box" software that hides the underlying mathematics, this application is designed as a **"Glass-Box" educational tool**. It bridges the gap between finite element theory and computational execution, allowing engineering students to observe the formulation of $6 \times 6$ local stiffness matrices and 3D direction cosines in real-time.

## ✨ Key Features

* 🎓 **Educational "Glass-Box" Engine:** View step-by-step mathematical formulations including 3D direction cosines ($l, m, n$), $6 \times 6$ element stiffness matrices ($k$), and matrix partitioning.
* 🌐 **Interactive 3D Visualization:** Renders high-fidelity, interactive Plotly 3D graphics displaying undeformed geometry, load application, and color-coded axial forces (Tension/Compression).
* 🔄 **Real-Time Unit Scaling:** Seamlessly toggle the visual output between Newtons (N), Kilonewtons (kN), and Meganewtons (MN) without altering the base SI solver engine.
* 📝 **1-Click Professional Reporting:** Automatically generates a comprehensive `.docx` calculation report containing embedded 3D graphics, nodal displacements, and categorized member forces.

## 🚀 Installation & Local Setup

### Prerequisites
* Python 3.9 or newer.

### Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mdaipayan/Truss_app.git](https://github.com/mdaipayan/Truss_app.git)
   cd Truss_app
