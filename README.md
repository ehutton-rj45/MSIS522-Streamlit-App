# MSIS 522 Data Science Workflow - Kitsap Rainfall

This repository is for the Homework 1: Data Science Workflow assignment. The dataset analyzes historical rainfall aggregates across Kitsap County, and predicting the next day's average regional rainfall.

## Key Files
- `app.py`: The Main Streamlit Dashboard containing Part 1, Part 3, and Part 4 representations.
- `hw1_modeling.py`: The analysis script containing Part 2 (model generation, CV tuning) logic.
- `requirements.txt`: To recreate the environment.

## How to use
1. Install requirements
2. Run `python hw1_modeling.py` to regenerate the models from the initial `.csv` data if necessary (all models are already pre-trained and saved anyways).
3. Run `streamlit run app.py` to host the web application locally.

## Models Evaluated
The assignment trained and compared the following models:
- Linear Regression (Baseline)
- Decision Tree
- Random Forest
- LightGBM (Gradient Boosted Trees)
- Multi-Layer Perceptron (Neural Network using PyTorch)
- Two-Stage LGBM + Tweedie (Bonus Original Model)
