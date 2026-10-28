# Insurance Cost Prediction

Predicts a person's medical insurance charges from age, sex, BMI, number of children, smoker status and region, using scikit-learn regression models.

## Dataset
- 1,338 records, reduced to 1,337 after removing one duplicate
- Features: `age`, `sex`, `bmi`, `children`, `smoker`, `region`
- Target: `charges` (a continuous dollar amount, so this is a regression problem)
- No missing values; smokers are about 20% of the data

## Approach
1. **Data checks:** missing values, duplicates, category counts and value ranges
2. **EDA:** univariate and bivariate charts with a written observation under each one
3. **Outlier check:** IQR limits computed on the training set; outliers were kept because 97% of the high charges belong to smokers
4. **Encoding:** one-hot encoding with `pd.get_dummies(drop_first=True)` (8 input columns)
5. **Models:** Linear Regression, Decision Tree, Random Forest, Gradient Boosting
6. **Evaluation:** 5-fold cross-validation on the training set using R², MAE and RMSE
7. **Tuning:** `GridSearchCV` on the two best models
8. **Final test:** the held-out test set was scored once, after the model was chosen

## Results (5-fold cross-validation)

| Model | CV R² | MAE | RMSE |
|-------|-------|-----|------|
| Linear Regression | 0.726 | 4,207 | 6,114 |
| Random Forest | 0.820 | 2,790 | 4,956 |
| Decision Tree | 0.833 | 2,736 | 4,771 |
| Gradient Boosting | 0.834 | 2,646 | 4,757 |
| Decision Tree (tuned) | 0.834 | 2,705 | 4,760 |
| **Gradient Boosting (tuned)** | **0.843** | **2,613** | **4,625** |

**Final model:** tuned Gradient Boosting (learning rate 0.05, max depth 2, 200 trees).
**Test set:** R² 0.899, MAE 2,523, RMSE 4,299 (268 rows, so scores are noisy; the cross-validated R² of about 0.84 is the more cautious figure).

## Key findings
- Smoking is by far the strongest driver of charges (feature importance 0.695), followed by BMI (0.174) and age (0.120). Sex, region and children contribute almost nothing.
- BMI raises charges mainly for smokers, which a plain linear model cannot capture. Tree-based models handle it better.
- Random Forest overfits (training R² 0.974 vs cross-validation R² 0.820).

## Limitations
- The model underpredicts the most expensive patients (bills above about 50,000 are predicted at about 37,000 to 46,000).
- The dataset is small and has no medical history.
- Capping versus keeping outliers was decided from the EDA, not tested experimentally.
- This is a learning project, not a pricing tool.

## How to run
```
pip install -r requirements.txt
jupyter notebook insurance_cost_prediction.ipynb
```

## Files
- `insurance_cost_prediction.ipynb`: full analysis, models and report on the challenges faced
- `insurance.csv`: dataset
- `requirements.txt`: Python dependencies
