# BigMart Sales Prediction

## Brief overview of the project

This project predicts `Item_Outlet_Sales` for products sold across
BigMart outlets using item-level and outlet-level characteristics.

The problem is treated as a **regression task**, with **RMSE** as the
primary evaluation metric and MAE and R² reported alongside it.

The project focuses on leakage-safe preprocessing, feature engineering,
model comparison, hyperparameter tuning, residual diagnostics, and model
serialization.

## Methodology

1.  Loaded the BigMart dataset containing **8,523 rows and 12 columns**.
2.  Performed data-quality checks and identified missing values in
    `Item_Weight` and `Outlet_Size`, inconsistent labels in
    `Item_Fat_Content`, and suspicious zero values in `Item_Visibility`.
    No duplicate rows were found.
3.  Standardized the inconsistent `Item_Fat_Content` categories.
4.  Engineered:
    -   `Item_Category` from the first two characters of
        `Item_Identifier`
    -   `Outlet_Age` using 2013 as the reference year
    -   `Item_Visibility` based on a training-set cross-validation
        comparison of alternative treatments
5.  Removed `Item_Identifier` and `Outlet_Establishment_Year` after
    feature engineering.
6.  Performed an **80/20 train-test split before fitting preprocessing
    steps** to prevent data leakage.
7.  Used grouped-mode imputation for missing `Outlet_Size`, based on
    `Outlet_Type`.
8.  Used a `ColumnTransformer` for numeric imputation/scaling and
    categorical imputation/one-hot encoding.
9.  Built **Linear Regression** as an interpretable baseline and checked
    its assumptions, including linearity, homoscedasticity, residual
    normality, and multicollinearity using VIF.
10. Compared three non-linear ensemble models --- **Random Forest,
    Gradient Boosting, and XGBoost** --- against the Linear Regression
    baseline using **5-fold cross-validation RMSE**.
11. Selected **Gradient Boosting** based on the lowest mean CV RMSE.
12. Tuned the selected model using `RandomizedSearchCV` using only the
    training data.
13. Evaluated the locked final model once on the held-out test set.
14. Performed post-hoc residual analysis and permutation feature
    importance on the test set.
15. Saved the complete preprocessing and Gradient Boosting pipeline
    using `joblib`.

## Model Comparison

  Model                     Mean CV RMSE
  ----------------------- --------------
  **Gradient Boosting**      **1,096.1**
  Random Forest                  1,157.9
  Linear Regression                  ---
  XGBoost                        1,199.3

Gradient Boosting achieved the lowest mean cross-validated RMSE and was
therefore selected for hyperparameter tuning.

## Hyperparameter Tuning

`RandomizedSearchCV` was used to tune the Gradient Boosting model.

Best configuration:

``` text
learning_rate = 0.03
n_estimators = 200
max_depth = 3
min_samples_leaf = 1
subsample = 0.7
```

Tuned CV RMSE:

**1,094.2**

## Final Test Results

The final locked Gradient Boosting model was evaluated on the untouched
test set.

  Metric             Score
  ---------- -------------
  **RMSE**     **1,027.9**
  **MAE**        **719.4**
  **R²**        **0.6113**

The test set was used only after model selection and hyperparameter
tuning were completed.

## Model Diagnostics

Residual analysis showed:

-   Mean residual ≈ **-21.2**, indicating little overall systematic
    bias.
-   A visible **funnel pattern**, with errors increasing for higher
    predicted sales.
-   A heavier-tailed residual distribution, indicating that larger
    prediction errors are more likely for high-sales observations.

## Feature Importance

**Permutation importance** was calculated on the held-out test set as a
post-hoc interpretation method.

It was not used for model selection, tuning, or changing the final
model.

## Model Saving

The complete trained pipeline is saved using `joblib`:

``` python
joblib.dump(best_model, "bigmart_sales_pipeline.pkl")
```

The saved pipeline includes preprocessing and the tuned Gradient
Boosting model.

The model expects the **engineered input format** used during training,
including `Item_Category` and `Outlet_Age`, with `Item_Identifier` and
`Outlet_Establishment_Year` already removed.

## Tech Stack

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   XGBoost
-   Matplotlib
-   Seaborn
-   Joblib
-   Jupyter Notebook

## Key Takeaways

-   `Item_MRP` showed the strongest visible numeric relationship with
    sales.
-   Outlet type showed substantial differences in sales.
-   Linear Regression provided an interpretable baseline but showed
    limitations related to linearity and multicollinearity.
-   Tree-based ensemble models were better suited to the non-linear
    structure of the problem.
-   **Gradient Boosting** achieved the best cross-validated performance
    and was selected as the final model.
-   Leakage-safe preprocessing and training-only cross-validation were
    used throughout model selection and tuning.
