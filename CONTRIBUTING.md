# Contributing

Thank you for your interest in contributing!

## Getting started

1. Fork the repository and clone your fork
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Set up the environment: `conda env create -f environment.yml`
4. Make your changes
5. Push your branch and open a Pull Request against `main`

## Guidelines

- **Notebooks:** Keep notebooks clean — clear all cell outputs before committing (`Kernel → Restart & Clear Output`)
- **Data files:** Never commit raw CSV data or model checkpoints (`.gitignore` covers these)
- **Dependencies:** If you add a new dependency, update both `requirements.txt` and `environment.yml`
- **XGBoost version:** Do not upgrade XGBoost past 2.1.1 — SHAP 0.49.x is incompatible with 3.x

## Reporting issues

Open a GitHub Issue with:
- Your OS and Python version
- The full error traceback
- Which notebook and step you were running
