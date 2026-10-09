# Project Report: Smart Product Recommendation System

**Author:** First-Year Computer Science / Engineering Student  
**Project Folder:** `nexus-project`  
**Application URL:** [http://localhost:8501](http://localhost:8501)  
**Technologies:** Python, Scikit-Learn, Pandas, NumPy, Streamlit, Plotly  

---

## 1. Problem Statement

In modern e-commerce platforms, customers are overwhelmed by catalogs containing thousands of products. Showing generic or unranked items results in high bounce rates, low user engagement, and lost revenue. 

The goal of this project is to build an intelligent, real-time **Smart Product Recommendation Engine** that:
1. Predicts the likelihood (probability between 0% and 100%) that a specific shopper will purchase a given product based on browsing behavior and past loyalty.
2. Ranks all available products and recommends the **Top 5 personalized items** with explainable reasons.
3. Automatically adapts to both established shoppers and brand-new visitors (**cold-start problem**).
4. Provides an interactive web application that allows non-technical business users to test and explore the system easily.

---

## 2. Methodology & Approach

The system is organized into a modular pipeline following industry best practices:

```
[Raw Interactions] ➔ [Data Cleaning] ➔ [Feature Engineering] ➔ [Model Training & CV] ➔ [Interactive UI]
```

### A. Realistic Synthetic Data Generation (`generate_data.py`)
Because production e-commerce clickstream data is proprietary and protected by privacy laws, we developed a synthetic generation script producing **5,150 customer session records** across 400 unique customers and 50 products in 6 categories (*Electronics, Clothing, Home & Kitchen, Books, Beauty, Sports*).

To mirror authentic web data, we modeled realistic shopping signals:
- **Category Affinity:** Shoppers preferentially browse in their favorite categories.
- **Cart Intent:** Cart additions serve as a decisive intent signal compared to passive views.
- **Real-World Flaws Injected:** Deliberately introduced **130 duplicate rows** (simulating accidental double-clicks) and **~4% missing values** across prices, views, cart additions, and categories.

### B. Automated Data Cleaning Pipeline (`train_model.py`)
1. **Deduplication:** Dropped 130 duplicate rows, yielding 5,020 clean interaction records.
2. **Missing Value Imputation:**
   - **Numerical Features** (`product_price`, `views`, `cart_adds`, `previous_purchases`): Imputed using the **median** (e.g. median price = \$60.35). The median was selected because it is resistant to extreme outliers like luxury goods.
   - **Categorical Features** (`product_category`): Imputed using the **mode** (most frequent category: `Books`).
3. **Logical Integrity:** Capped `cart_adds` so it never exceeds total `views`.

### C. Feature Engineering (Strict Zero Data Leakage)
A common beginner error is **data leakage**—using information during training that wouldn't be available at prediction time. In our pipeline:
- The target variable is `will_purchase = 1` if `purchases > 0`, otherwise `0`.
- The raw `purchases` column was **strictly excluded** from the input features ($X$).

Engineered features created:
- `cart_rate = cart_adds / views`: Measures the customer's conversion intent.
- `cart_intent_ratio = (cart_adds * 2) / (views + 1)`: Balances cart adds against browsing depth.
- `loyalty_score = previous_purchases / (previous_purchases + 5)`: Diminishing returns formula reflecting repeat buyer trust.
- `price_bucket`: Binned into *Budget* (<$50), *Mid-Range* ($50–$150), *Premium* ($150–$400), and *Luxury* (>$400).
- `price_per_view = product_price / (views + 1)`: Ratio of cost to user attention.

### D. Models Evaluated and Rationale
We benchmarked three distinct algorithms using stratified splits (80% train, 20% test):
1. **Logistic Regression (Baseline):** A linear classifier that outputs calibrated probabilities and interpretable feature weights.
2. **Random Forest Classifier:** An ensemble of decision trees that handles non-linear relationships and feature interactions without overfitting.
3. **Gradient Boosting Classifier:** A sequentially boosted ensemble that minimizes residual errors iteratively.

---

## 3. Evaluation Results & Benchmarks

All models were evaluated using **5-Fold Stratified Cross-Validation** on the training data, followed by **GridSearchCV hyperparameter tuning** on the top performer.

### 5-Fold Stratified Cross-Validation Leaderboard

| Model | CV Accuracy (Mean ± Std) | CV F1 Score (Mean ± Std) | CV ROC-AUC (Mean ± Std) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** 🏆 | **77.54% ± 1.23%** | **81.54% ± 1.00%** | **0.8444 ± 0.0128** |
| **Random Forest** | 77.14% ± 1.03% | 81.48% ± 0.91% | 0.8441 ± 0.0141 |
| **Gradient Boosting** | 76.94% ± 1.08% | 81.27% ± 0.92% | 0.8379 ± 0.0145 |

### Holdout Test Set Performance (Final Tuned Model)

The top base model was tuned using `GridSearchCV` (evaluating regularization strengths $C \in \{0.1, 1.0, 5.0\}$ and solver iterations). The optimal parameters were found to be **$C = 1.0$, $\text{max\_iter} = 500$**.

- **Accuracy:** **77.29%** (Overall correct classification rate)
- **Precision:** **78.21%** (Of all predicted buyers, 78.2% actually bought)
- **Recall (Sensitivity):** **85.33%** (Caught 85.3% of all actual purchase conversions)
- **F1 Score:** **81.61%** (Harmonic mean of precision and recall)
- **ROC-AUC:** **0.8364** (Area under the ROC curve, measuring probability calibration)

### Key Feature Drivers
The model relies most heavily on:
1. `cart_rate` (strongest predictor of conversion)
2. `cart_intent_ratio`
3. `cart_adds`
4. `loyalty_score` (repeat customer history)

---

## 4. How the Top 5 Recommendation System Works

When a customer profile is selected or simulated:
1. **Catalog Scoring:** The model evaluates all 50 catalog items against the customer's behavioral state and calculates a purchase probability for each product.
2. **Exclusion of Previously Purchased Products:**
   - In e-commerce, recommending products a user already owns leads to poor user experience.
   - The system queries `customer_purchases` and filters out previously bought items before ranking, ensuring 100% novel suggestions. An optional toggle allows disabling this filter if re-purchases are desired.
3. **Cold-Start Handling for New Customers:**
   - New visitors have 0 previous orders.
   - The engine automatically detects cold-start status and relies on **real-time session engagement** (current page views, cart additions, and category preferences) combined with baseline catalog conversion probabilities.
   - An explanatory banner in the UI informs the user that cold-start discovery logic is active.
4. **Ranking & "Why Recommended" Explanations:**
   - The top 5 highest-probability items are displayed with visual cards and progress indicators.
   - Each product is accompanied by a plain-English explanation generated dynamically (e.g., *"High cart-add interest (75% conversion rate)"*, *"Matches preferred category (Electronics)"*).
5. **CSV Export:** Users can click **"Download Recommendations as CSV"** to save recommendations for downstream marketing or email campaigns.

---

## 5. How to Run the Project

### Prerequisites
- Python 3.10+ installed on Windows, macOS, or Linux.

### Setup and Execution Commands
```powershell
# 1. Navigate to the project directory
cd nexus-project

# 2. Activate the virtual environment
.venv\Scripts\activate

# 3. (Optional) Regenerate the dataset and train models from scratch
python generate_data.py
python train_model.py

# 4. Launch the Streamlit Web Application
streamlit run app.py
```
Open your browser and navigate to: **[http://localhost:8501](http://localhost:8501)**.

---

## 6. Limitations & Future Improvements

### Limitations
1. **Synthetic Dataset:** While our generation function models authentic statistical properties (cart rates, category loyalty, missing values), synthetic data cannot capture complex real-world behavioral nuances like seasonality, holiday promotions, or supply chain stockouts.
2. **Catalog Size:** The catalog is currently configured with 50 items. In enterprise production, catalogs contain millions of SKUs, requiring a two-stage pipeline (candidate retrieval followed by heavy ranking).

### Future Improvements
1. **Collaborative Filtering & Embeddings:** Integrate matrix factorization (e.g., SVD or ALS) or two-tower neural network embeddings (Word2Vec / Item2Vec) to learn latent similarities between products.
2. **Session-Based Recurrent Models:** Implement sequential models (GRUs or Transformers) to predict the next click based on the exact click order in a browsing session.
3. **A/B Testing Framework:** Deploy the system into a live e-commerce test harness to measure real-world conversion lift, click-through rate (CTR), and average order value (AOV).

---

## 7. Conclusion

This project successfully demonstrates an end-to-end Machine Learning solution for e-commerce product recommendations. By emphasizing data cleaning, preventing data leakage, validating performance across 5-fold cross-validation, and packaging the result into an intuitive, dark-mode compatible Streamlit dashboard, the system bridges theoretical ML with practical application.
