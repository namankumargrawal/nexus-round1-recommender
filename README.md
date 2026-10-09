# Smart Product Recommendation System

A complete, beginner-friendly Machine Learning recommendation system built in Python, Scikit-Learn, and Streamlit.

---

## 🌟 What Does This Project Do?

This application predicts the likelihood that an e-commerce customer will purchase a product based on real-time browsing behavior, category preferences, and past loyalty. It uses this predicted purchase probability to rank and recommend the **Top 5 Products** in real-time.

### Upgraded Key Features:
1. **Realistic High-Signal Synthetic Dataset** (`generate_data.py`):
   - Generates ~5,000 shopper sessions with intentional duplicate clicks (~3%) and missing values (~4%) across numerical and categorical features.
   - Models realistic conversion dynamics: strong cart-to-view conversion signal, category preferences, repeat buyer loyalty, and price tiers.
2. **Automated Data Cleaning & Imputation** (`train_model.py`):
   - Automatically drops duplicate interactions.
   - Imputes missing numeric values using the **median** (outlier-resistant) and missing categories using the **mode** (`Books`).
3. **Feature Engineering (Strict Zero Data Leakage)**:
   - Derives `cart_rate`, `cart_intent_ratio`, `loyalty_score`, `price_bucket`, and `price_per_view`.
   - The raw `purchases` column is strictly excluded from feature matrix $X$ and is only used to create the target label `will_purchase`.
4. **5-Fold Cross-Validation & Light Hyperparameter Tuning (GridSearchCV)**:
   - Evaluates **Logistic Regression**, **Random Forest**, and **Gradient Boosting** using 5-fold stratified cross-validation.
   - Automatically tunes hyperparameters with `GridSearchCV` on the best base model to maximize **ROC-AUC**.
5. **Interactive Model Comparison & Visualizations**:
   - Interactive **ROC Curve** comparison chart with AUC scores.
   - Interactive **Confusion Matrix** heatmap displaying True Positives, True Negatives, False Positives, and False Negatives.
6. **Smart Recommendation Filtering**:
   - **Excludes Previously Purchased Products**: Customers are not recommended products they have already bought (with an on/off toggle).
7. **Cold-Start Handling for New Customers**:
   - Gracefully handles first-time visitors with no prior purchase history, dynamically recommending top discovery products using session intent and category preference.
8. **Exportable Results**:
   - One-click **"Download Recommendations as CSV"** button.

---

## 🚀 How to Run

### Step 1: Activate the Virtual Environment
Open your terminal in this project folder (`nexus-project`):
```powershell
.venv\Scripts\activate
```

### Step 2: (Optional) Regenerate Data and Retrain Models
To run the pipeline from scratch via the terminal:
```powershell
.venv\Scripts\python.exe generate_data.py
.venv\Scripts\python.exe train_model.py
```

### Step 3: Launch the Streamlit Web Application
```powershell
.venv\Scripts\streamlit.exe run app.py
```
Open your browser and navigate to: **[http://localhost:8501](http://localhost:8501)**

---

## 📚 Key Concepts for Your Presentation / Viva

### 1. What is Data Leakage and How Did We Prevent It?
- **Data leakage** happens when information from outside the training dataset (or the future outcome itself) is used to create features.
- If we included `purchases` as an input feature to predict whether someone will purchase, the model would cheat and learn nothing useful.
- **Our solution**: We strictly excluded `purchases` from the input features ($X$). `purchases` was solely used to define the target label `will_purchase` (1 if purchases > 0, else 0).

### 2. Why Did We Use Median and Mode for Missing Values?
- **Median** is used for numerical features (like `views` and `product_price`) because it is robust against extreme outliers (a few very expensive luxury items won't skew the median).
- **Mode** (the most frequent value) is the standard method for categorical features (like `product_category`).

### 3. What is 5-Fold Cross-Validation and Why Use It?
- Instead of relying on a single train-test split, 5-Fold Cross-Validation splits the training data into 5 equal subsets (folds). The model is trained on 4 folds and tested on the 5th, repeated 5 times.
- This ensures the model's performance is stable, consistent, and not just lucky on a specific split.

### 4. What is Hyperparameter Tuning (GridSearchCV)?
- Machine learning models have configuration settings called **hyperparameters** (like regularization strength $C$ or the number of trees $n\_estimators$).
- `GridSearchCV` tests combinations of these parameters across cross-validation folds to mathematically identify the optimal settings.

### 5. What are the ROC Curve and ROC-AUC?
- The **ROC Curve** (Receiver Operating Characteristic) plots the **True Positive Rate** (Sensitivity/Recall) against the **False Positive Rate** at all possible classification thresholds.
- **ROC-AUC** measures the area under this curve (ranging from 0.5 for random guessing to 1.0 for a perfect model). A higher ROC-AUC means the model outputs well-calibrated probabilities for ranking top recommendations.

### 6. How is Cold Start Handled?
- When a new customer arrives without any historical order logs, the system relies on real-time session signals (browsing views, cart additions, and selected category interest) combined with catalog conversion rates to recommend products immediately.

---

## 📁 Project Structure

```
nexus-project/
├── data.csv                 # Raw dataset with duplicates & missing values (~5,150 rows)
├── generate_data.py         # Generates realistic synthetic e-commerce data
├── train_model.py           # Cleans data, engineers features, runs 5-fold CV & tuning
├── app.py                   # Streamlit interactive web application
├── model_artifacts.pkl      # Pickled pipeline, preprocessor, metrics, and mappings
├── requirements.txt         # Required Python packages
└── README.md                # Documentation guide and viva prep
```
