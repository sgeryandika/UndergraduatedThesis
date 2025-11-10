# UndergraduatedThesis – Maintenance Packaging Problem

This repository contains the code and materials developed for my undergraduate thesis on the **Maintenance Tasks Packaging Problem**.  
The project explores how to optimize maintenance tasks packaging using **heuristic baselines** and **Mixed-Integer Linear Programming (MILP)** models, with preprocessing and frontend components to support realistic workflows.

---

## 📂 Repository Structure

- **[Thesis_Draft(Final).pdf](./Thesis_Draft(Final).pdf)**  
  Final draft of the thesis document, including background, methodology, and results.

- **[mtp_preprocessing](./mtp_preprocessing/)**  
  Scripts and data preprocessing utilities. These prepare input data for optimization models.

- **[mtp_heuristic_baseline](./mtp_heuristic_baseline/)**  
  Heuristic algorithms used as baseline comparisons. Provides fast but non-optimal solutions.

- **[mtp_milp_optimizer](./mtp_milp_optimizer/)**  
  MILP-based optimizer implementation. Finds provably optimal solutions under linear surrogate models.

- **[mtp_frontend](./mtp_frontend/)**  
  Frontend scripts for running experiments and interacting with preprocessing + optimizers.

- **[LICENSE](./LICENSE)**  
  MIT License — free to use, modify, and distribute with attribution.

---

## 🚀 How to Use

1. **Preprocess data**  
   Run scripts in `mtp_preprocessing` to generate input files.

2. **Run heuristic baseline**  
   Execute code in `mtp_heuristic_baseline` to produce quick schedules.

3. **Run MILP optimizer**  
   Use `mtp_milp_optimizer` to compute optimal schedules under surrogate models.

4. **Frontend interaction**  
   `mtp_frontend` provides a simple interface to combine preprocessing and optimization runs.

---

## 🎯 Research Goals

- Compare heuristic vs. MILP approaches for maintenance packaging.  
- Provide reproducible experiments for academic and practical use.

---

## 🧩 Notes on Code Generation

Parts of the code were generated with **AI assistance** based on my design and specifications.  
The originality lies in the **conceptual design, problem formulation, and integration strategy**, while AI tools accelerated coding tasks.  
All code was reviewed and adapted to ensure correctness.

---

## 📜 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute the code, provided that you retain the license and attribution.

---

## 🙌 Acknowledgements

- Supervisors and peers who provided feedback on the thesis.  
- Open-source solver libraries and AI-assisted tools that supported implementation.  
