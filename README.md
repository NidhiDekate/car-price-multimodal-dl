# Car Price Prediction — Multimodal Deep Learning

Predicting used car auction prices using both structured vehicle attributes and vehicle images.

🚗 Live Demo: https://huggingface.co/spaces/NidhiDekate/car-price-prediction

---

## Overview

The goal of this project was to determine whether vehicle images could improve price prediction beyond traditional tabular machine learning models.

To test this, I built and compared:

- XGBoost using tabular vehicle attributes
- ResNet18 using vehicle images only
- A multimodal architecture combining image and tabular features

The result was unexpected: the tabular model significantly outperformed both image-only and multimodal approaches.

Understanding why this happened became the most valuable outcome of the project.

---

## Results

| Model | RMSE | R² |
|---------|---------|---------|
| XGBoost (tabular features) | $1,872 | 0.95 |
| ResNet18 (images only) | $10,454 | -0.49 |
| Multimodal v1 (frozen ResNet18) | $8,772 | -0.10 |
| Multimodal v2 (unfrozen layer4) | $7,892 | 0.11 |

### Key Observation

Vehicle auction prices are primarily driven by:

- Mileage (odometer)
- Vehicle age
- Brand tier
- Trim level
- Condition score

Most of these factors are not directly observable from a single vehicle image.

Although the CNN learned meaningful visual representations, the image branch lacked access to many of the variables responsible for price variation. As a result, adding image features did not improve performance over the tabular baseline.

This project reinforced an important machine learning principle:

> Model performance is limited by the predictive signal available in the data.

---

## Dataset

### Vehicle Sales Data

- 558,000 vehicle auction records
- Features include:
  - Make
  - Model
  - Trim
  - Body Style
  - Odometer
  - Condition Score
  - State
  - Exterior Color
  - Interior Color
  - Selling Price

### Vehicle Images

- Stanford Cars Dataset
- 8,144 training images
- 196 vehicle classes

### Matched Dataset

To create a multimodal dataset, vehicle records were aligned with Stanford Cars classes, producing:

- 40,658 image-tabular pairs

A limitation of this approach is that multiple sales records can share the same representative image, reducing the amount of pricing information available to the image branch.

---

## Project Structure

| Notebook | Purpose |
|----------|----------|
| `01_data_setup.ipynb` | Data loading, Stanford Cars matching, multimodal dataset creation |
| `02_EDA.ipynb` | Exploratory analysis and feature investigation |
| `03_preprocessing.ipynb` | Cleaning, encoding, scaling, train/validation/test splits |
| `04_ML_models.ipynb` | Ridge, CatBoost, LightGBM, XGBoost, and stacking experiments |
| `05_CNN_branch.ipynb` | ResNet18 image regression model |
| `06_multimodal_fusion.ipynb` | CNN + ANN late-fusion architecture |
| `07_explainability.ipynb` | SHAP and Grad-CAM analysis |
| `08_deployment.ipynb` | Gradio application deployment |

---

## Model Architecture

### Image Branch

- ResNet18 pretrained on ImageNet
- Final classification layer replaced with a regression head
- Outputs a 512-dimensional image embedding

### Tabular Branch

Input features include:

- Vehicle age
- Odometer
- Condition score
- Make
- Model
- Trim
- Body style
- State
- Exterior color
- Interior color

Architecture:

```text
12 → 64 → 32
```

### Fusion Layer

```text
Image Features (512)
          │
          ▼
      Concatenate ──► 544
          ▲
          │
Tabular Features (32)

544 → 256 → 64 → 1
```

Fusion Strategy:

- Late fusion
- Independent feature extraction for each modality
- Joint regression head for final prediction

---

## Explainability

### SHAP Feature Importance

Top predictors of vehicle price:

1. Brand tier
2. Vehicle age
3. Log-transformed odometer reading
4. Model / Trim
5. Condition score

Transmission type, exterior color, and interior color contributed very little to model predictions.

### Grad-CAM Analysis

Grad-CAM visualizations showed that the CNN consistently focused on:

- Vehicle body panels
- Front grille
- Wheels

rather than background regions.

This indicates that the model learned meaningful vehicle-centric representations despite poor predictive performance.

---

## Technology Stack

- Python
- PyTorch
- ResNet18
- XGBoost
- LightGBM
- CatBoost
- Scikit-Learn
- SHAP
- Grad-CAM
- Gradio
- Hugging Face Spaces
- Google Colab

---

## Lessons Learned

This project started with the assumption that combining images and structured data would improve performance.

The experiments showed otherwise.

The strongest predictors remained structured vehicle attributes, while images contributed little additional information because the dataset lacked vehicle-specific condition details.

If unique listing photos were available for every vehicle, the image branch could potentially learn visual indicators such as:

- Paint damage
- Body wear
- Dents and scratches
- Modifications
- Tire condition

In that setting, multimodal learning would likely provide meaningful gains over tabular models alone.

---

## Author

**Nidhi Dekate**

MSAI — The University of Texas at Austin

Former Frontend Engineer transitioning into Machine Learning and Applied AI.

Interested in building end-to-end ML systems, explainable AI, and production-ready machine learning applications.
