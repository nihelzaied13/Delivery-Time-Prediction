# Delivery Time Prediction

Predict the delivery time (in minutes) of orders based on driver characteristics, order details, and distance traveled. This repository walks through the full machine learning pipeline—from data loading and preprocessing to model training, evaluation, and inference via a serialized pipeline.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Environment & Installation](#environment--installation)
4. [Notebook Tutorial](#notebook-tutorial)
5. [Modeling & Results](#modeling--results)
6. [Feature Importance](#feature-importance)
7. [Inference Example](#inference-example)
8. [Project Structure](#project-structure)
9. [Contributing](#contributing)
10. [License](#license)
11. [Author](#author)

---

## Project Overview

This project aims to predict the delivery time of an order (in minutes) using features such as:

- **Delivery_person_Age**: Age of the delivery person
- **Delivery_person_Ratings**: Rating score of the delivery person
- **distance_km**: Distance of the route in kilometers
- **Type_of_order**: Category of the order (e.g., Snack, Meal)
- **Type_of_vehicle**: Vehicle used for delivery (e.g., motorcycle, bike)

Key steps:

1. Data loading and exploratory analysis  
2. Cleaning and preprocessing (handling missing values, encoding categoricals)  
3. Feature engineering and scaling  
4. Training multiple regressors (Linear Regression, Random Forest, XGBoost)  
5. Hyperparameter tuning with `GridSearchCV`  
6. Building a stacking ensemble  
7. Analyzing feature importances  
8. Serializing the best pipeline for production use

---

## Dataset

Download or place the **`deliverytime.csv`** file in the project root. It should contain at least the following columns:

| Column                  | Type         | Description                              |
|-------------------------|--------------|------------------------------------------|
| Delivery_person_Age     | float        | Age of the delivery person               |
| Delivery_person_Ratings | float        | Rating score of the delivery person      |
| distance_km             | float        | Distance of the delivery route           |
| Type_of_order           | categorical  | Category of the order                    |
| Type_of_vehicle         | categorical  | Vehicle type used by the delivery person |
| Time_taken(min)         | float        | Target variable: delivery time (min)     |

---

## Environment & Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd <repo-directory>

# (Optional) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt** should include:
```
pandas
numpy
matplotlib
scikit-learn
xgboost
joblib
```

---

## Notebook Tutorial

All steps are demonstrated in the Jupyter notebook **`delivery_time_prediction.ipynb`**:

1. **Import libraries**  
2. **Load & explore data**  
3. **Clean & preprocess**  
4. **Feature engineering**  
5. **Exploratory Data Analysis (EDA)**  
6. **Modeling & initial evaluation**  
7. **Interpretation (feature importances)**  
8. **Save the best model pipeline**  
9. **Inference example**  

Run:
```bash
jupyter notebook delivery_time_prediction.ipynb
```

---

## Modeling & Results

| Model                          | MAE    | RMSE   |  R²    |
|--------------------------------|--------|--------|--------|
| Linear Regression (baseline)   | 6.6915 | 8.4318 | 0.2145 |
| Random Forest (default)        | 6.0769 | 7.7801 | 0.3313 |
| Random Forest (tuned)          | 5.7537 | 7.3261 | 0.4070 |
| XGBoost (default)              | 5.8607 | 7.4721 | 0.3832 |
| XGBoost (tuned)                | 5.7393 | 7.2926 | 0.4124 |
| Stacking Ensemble              | 5.7292 | 7.2842 | 0.4138 |

The stacking ensemble (combining tuned Random Forest and XGBoost) achieved the best performance.

---

## Feature Importance

The optimized XGBoost model’s feature importances are plotted in the notebook under the **Interpretation** section, showing which features contribute most to predicting delivery time.

---

## Inference Example

```python
import joblib
import pandas as pd

# Load the serialized pipeline
pipeline = joblib.load('delivery_xgb_pipeline.joblib')

# Create a new sample for prediction
new_sample = pd.DataFrame([{
    'Delivery_person_Age': 37.0,
    'Delivery_person_Ratings': 4.9,
    'distance_km': 3.025149,
    'Type_of_order': 'Snack',
    'Type_of_vehicle': 'motorcycle'
}])

# Predict delivery time
predicted_time = pipeline.predict(new_sample)[0]
print(f"Estimated delivery time: {predicted_time:.2f} minutes")
```

---

## Project Structure

```
├── delivery_time_prediction.ipynb  # Jupyter notebook
├── deliverytime.csv               # Dataset file
├── delivery_xgb_pipeline.joblib    # Serialized XGBoost pipeline
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository  
2. Create a new branch (`feature/YourFeature`)  
3. Commit your changes  
4. Open a Pull Request  

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Author

**Nihel Zaied**  
Engineering Graduate in Statistics & Data Science  
Email: zaiednihel03@gmail.com
