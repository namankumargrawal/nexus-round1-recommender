"""
generate_data.py
----------------
Step 1: Synthetic Data Generation for Smart Product Recommendation System

Upgraded Version:
- Models realistic e-commerce customer behavior with clear signals:
  1. Category Affinity: Each customer has a preferred shopping category.
  2. Cart Conversion Signal: High cart-to-view ratios strongly drive purchases.
  3. Customer Loyalty: Repeat buyers have higher baseline conversion.
  4. Price Sensitivity: Realistic budget friction for luxury goods.
- Deliberately retains real-world data imperfections:
  - Duplicate rows (~3%)
  - Missing values / NaNs (~4%) across numerical and categorical features.
"""

import numpy as np
import pandas as pd

def generate_ecommerce_data(num_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic e-commerce interaction dataset with
    strong, realistic signals for machine learning.
    """
    np.random.seed(random_state)

    # 1. Catalog Definition (50 products across 6 categories)
    customer_ids = [f"CUST_{i:04d}" for i in range(1001, 1401)]
    product_ids = [f"PROD_{i:03d}" for i in range(1, 51)]

    categories = [
        "Electronics",
        "Clothing",
        "Home & Kitchen",
        "Books",
        "Beauty",
        "Sports"
    ]

    category_price_ranges = {
        "Electronics": (75.0, 850.0),
        "Clothing": (20.0, 150.0),
        "Home & Kitchen": (30.0, 320.0),
        "Books": (12.0, 60.0),
        "Beauty": (15.0, 120.0),
        "Sports": (25.0, 230.0)
    }

    # Assign fixed metadata to products
    product_catalog = {}
    for pid in product_ids:
        cat = np.random.choice(categories)
        min_p, max_p = category_price_ranges[cat]
        base_price = round(float(np.random.uniform(min_p, max_p)), 2)
        product_catalog[pid] = {"category": cat, "price": base_price}

    # 2. Customer Profiles (Affinity & Loyalty)
    # Each customer has a favorite category and a historical purchase count
    customer_profiles = {}
    for cid in customer_ids:
        fav_cat = np.random.choice(categories)
        loyalty_orders = int(np.random.choice(
            [0, 1, 2, 3, 5, 8, 12, 20],
            p=[0.22, 0.22, 0.18, 0.14, 0.11, 0.07, 0.04, 0.02]
        ))
        customer_profiles[cid] = {
            "favorite_category": fav_cat,
            "previous_purchases": loyalty_orders
        }

    # 3. Generate Browsing & Purchase Interactions
    rows = []
    for _ in range(num_samples):
        cid = np.random.choice(customer_ids)
        c_prof = customer_profiles[cid]
        fav_cat = c_prof["favorite_category"]
        prev_purchases = c_prof["previous_purchases"]

        # Customers browse their favorite category ~60% of the time
        if np.random.rand() < 0.60:
            candidate_pids = [p for p, d in product_catalog.items() if d["category"] == fav_cat]
            pid = np.random.choice(candidate_pids)
        else:
            pid = np.random.choice(product_ids)

        cat = product_catalog[pid]["category"]
        price = product_catalog[pid]["price"]
        is_fav_cat = (cat == fav_cat)

        # Browsing activity: views
        base_views = np.random.geometric(p=0.18) + 1
        views = min(base_views + (3 if is_fav_cat else 0), 40)

        # Cart additions: strong signal based on category interest and price
        if is_fav_cat:
            cart_prob = 0.45 if price < 200 else 0.30
        else:
            cart_prob = 0.18 if price < 100 else 0.08

        # Frequent buyers add to cart more eagerly
        cart_prob += min(prev_purchases * 0.015, 0.15)
        cart_adds = np.random.binomial(n=views, p=min(cart_prob, 0.85))
        cart_adds = min(cart_adds, views)

        # High-signal conversion logic:
        cart_rate = cart_adds / max(1, views)

        if cart_adds == 0:
            purchase_chance = 0.02
        else:
            # Calibrated log-odds score based on behavioral factors
            # High cart rate, category match, and loyalty strongly increase purchase likelihood
            z_score = (
                4.2 * cart_rate
                + 1.2 * (cart_adds / (views + 1.0))
                + (0.8 if is_fav_cat else -0.4)
                + min(prev_purchases * 0.08, 0.8)
                - (0.6 if price > 350 else 0.0)
                - 1.8
            )
            # Sigmoid conversion probability
            purchase_chance = 1.0 / (1.0 + np.exp(-z_score))
            purchase_chance = np.clip(purchase_chance, 0.03, 0.97)

        will_buy = (np.random.rand() < purchase_chance)

        if will_buy:
            max_purchases = max(2, min(cart_adds + 1, 4))
            purchases = int(np.random.randint(1, max_purchases))
        else:
            purchases = 0

        rows.append({
            "customer_id": cid,
            "product_id": pid,
            "product_category": cat,
            "product_price": price,
            "views": views,
            "cart_adds": cart_adds,
            "purchases": purchases,
            "previous_purchases": prev_purchases
        })

    df = pd.DataFrame(rows)

    # 4. Deliberately inject duplicate rows (~3% of data)
    num_duplicates = int(num_samples * 0.03)
    duplicate_rows = df.sample(n=num_duplicates, random_state=random_state)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    # 5. Deliberately inject missing values / NaNs (~4% per selected column)
    nan_mask_price = np.random.rand(len(df)) < 0.04
    df.loc[nan_mask_price, "product_price"] = np.nan

    nan_mask_views = np.random.rand(len(df)) < 0.03
    df.loc[nan_mask_views, "views"] = np.nan

    nan_mask_cart = np.random.rand(len(df)) < 0.03
    df.loc[nan_mask_cart, "cart_adds"] = np.nan

    nan_mask_cat = np.random.rand(len(df)) < 0.03
    df.loc[nan_mask_cat, "product_category"] = np.nan

    # Shuffle the dataset
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df

if __name__ == "__main__":
    print("Generating updated high-signal synthetic dataset...")
    df = generate_ecommerce_data(num_samples=5000)
    output_filename = "data.csv"
    df.to_csv(output_filename, index=False)

    print(f"Dataset successfully created and saved to '{output_filename}'!")
    print(f"Total Rows: {len(df)}")
    print(f"Duplicate Rows Count: {df.duplicated().sum()}")
    print("\nMissing Values per column:")
    print(df.isnull().sum())
    print(f"\nOverall Purchase Rate: {(df['purchases'] > 0).mean() * 100:.2f}%")
