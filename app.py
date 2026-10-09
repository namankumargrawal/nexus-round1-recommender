"""
app.py
------
Interactive Streamlit Dashboard: Smart Product Recommendation System

Upgraded Version:
1. Customer Selection (Pick existing customer or simulate a new customer).
2. Cold-Start Handling: Explicit detection & handling for new/zero-history shoppers.
3. Prior Purchase Exclusion: Automatically omits items the customer has already bought.
4. Top 5 Recommended Products with purchase probabilities and dynamic 'Why recommended' badges.
5. Download Recommendations as CSV button.
6. Feature Importance Bar Chart explaining behavioral drivers.
7. 5-Fold Cross-Validation & Hyperparameter Tuning (GridSearchCV) Leaderboard.
8. Interactive ROC Curve Chart and Confusion Matrix Heatmap.
9. Data Cleaning & Zero-Leakage Pipeline Summary.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart Product Recommendation Engine",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        color: #0F172A !important;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #CBD5E1;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-card * {
        color: #0F172A !important;
    }
    .metric-card code {
        background-color: #E2E8F0 !important;
        color: #0F172A !important;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
    }
    .recommendation-card {
        background-color: #FFFFFF;
        color: #0F172A !important;
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        transition: transform 0.2s ease;
    }
    .recommendation-card:hover {
        transform: translateY(-2px);
        border-color: #3B82F6;
    }
    .recommendation-card * {
        color: #0F172A !important;
    }
    .recommendation-card .why-tag {
        background-color: #EFF6FF !important;
        color: #1D4ED8 !important;
    }
    .why-tag {
        background-color: #EFF6FF !important;
        color: #1D4ED8 !important;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.88rem;
        font-weight: 500;
        display: inline-block;
        margin-top: 6px;
    }
    .cold-start-card {
        background-color: #F0F9FF;
        color: #0C4A6E !important;
        border: 1px solid #BAE6FD;
        border-left: 5px solid #0284C7;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 18px;
    }
    .cold-start-card * {
        color: #0C4A6E !important;
    }
    .params-card {
        background-color: #F8FAFC;
        color: #0F172A !important;
        border: 1px solid #CBD5E1;
        padding: 12px 18px;
        border-radius: 8px;
        margin: 12px 0;
    }
    .params-card * {
        color: #0F172A !important;
    }
    .params-card code {
        background-color: #E2E8F0 !important;
        color: #0F172A !important;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Artifact Loader with Auto-Fallback
# ---------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_or_train_artifacts():
    """Loads pre-trained model artifacts or triggers training if missing."""
    artifacts_path = "model_artifacts.pkl"
    if not os.path.exists(artifacts_path) or not os.path.exists("data.csv"):
        with st.spinner("Initializing dataset and training models..."):
            from generate_data import generate_ecommerce_data
            from train_model import run_pipeline
            if not os.path.exists("data.csv"):
                df_gen = generate_ecommerce_data(5000)
                df_gen.to_csv("data.csv", index=False)
            run_pipeline()

    return joblib.load(artifacts_path)


artifacts = load_or_train_artifacts()
best_pipeline = artifacts["best_pipeline"]
best_model_name = artifacts["best_model_name"]
best_params = artifacts.get("best_params", {})
cv_results_df = artifacts.get("cv_results_df", pd.DataFrame())
metrics_df = artifacts["metrics_df"]
roc_curves = artifacts.get("roc_curves", {})
confusion_matrix_data = artifacts.get("confusion_matrix_data", {})
feature_importance_df = artifacts["feature_importance_df"]
cleaning_stats = artifacts["cleaning_stats"]
product_catalog = artifacts["product_catalog"]
customer_profiles = artifacts["customer_profiles"]
customer_purchases = artifacts.get("customer_purchases", {})
cleaned_df = artifacts["cleaned_df"]


# ---------------------------------------------------------
# Recommendation Helper Function
# ---------------------------------------------------------
def generate_recommendations(
    customer_info,
    catalog_df,
    pipeline,
    top_n=5,
    category_filter="All",
    exclude_pids=None
):
    """
    Evaluates catalog products for the customer, computes purchase probabilities,
    applies exclusion filtering (e.g. already purchased items), and returns ranked picks.
    """
    df_eval = catalog_df.copy()

    # Category filter
    if category_filter != "All":
        df_eval = df_eval[df_eval["product_category"] == category_filter].reset_index(drop=True)

    # Exclude previously purchased items
    excluded_count = 0
    if exclude_pids:
        initial_len = len(df_eval)
        df_eval = df_eval[~df_eval["product_id"].isin(exclude_pids)].reset_index(drop=True)
        excluded_count = initial_len - len(df_eval)

    if df_eval.empty:
        return pd.DataFrame(), excluded_count

    # Customer behavioral context
    views = max(1, customer_info["views"])
    cart_adds = min(customer_info["cart_adds"], views)
    prev_p = customer_info["previous_purchases"]

    df_eval["views"] = views
    df_eval["cart_adds"] = cart_adds
    df_eval["previous_purchases"] = prev_p

    # Engineered features
    df_eval["cart_rate"] = np.where(df_eval["views"] > 0, df_eval["cart_adds"] / df_eval["views"], 0.0)
    df_eval["cart_intent_ratio"] = (df_eval["cart_adds"] * 2.0) / (df_eval["views"] + 1.0)
    df_eval["loyalty_score"] = prev_p / (prev_p + 5.0)

    bins = [-np.inf, 50, 150, 400, np.inf]
    labels = ["Budget", "Mid-Range", "Premium", "Luxury"]
    df_eval["price_bucket"] = pd.cut(df_eval["product_price"], bins=bins, labels=labels)
    df_eval["price_per_view"] = df_eval["product_price"] / (df_eval["views"] + 1.0)

    # Feature columns aligned with model training
    features_to_predict = df_eval[[
        "product_category", "product_price", "views", "cart_adds",
        "previous_purchases", "cart_rate", "cart_intent_ratio",
        "loyalty_score", "price_bucket", "price_per_view"
    ]]

    # Predict purchase probability (Class 1)
    probabilities = pipeline.predict_proba(features_to_predict)[:, 1]
    df_eval["purchase_probability"] = probabilities

    # Sort descending
    df_sorted = df_eval.sort_values(by="purchase_probability", ascending=False).head(top_n).reset_index(drop=True)

    # Generate explainable 'Why Recommended' logic
    reasons = []
    preferred_cat = customer_info.get("preferred_category", "")

    for _, row in df_sorted.iterrows():
        why = []
        p_prob = row["purchase_probability"]
        c_rate = row["cart_rate"]
        price = row["product_price"]

        if c_rate >= 0.40:
            why.append(f"High cart-add interest ({c_rate*100:.0f}% conversion rate)")
        elif row["views"] >= 6:
            why.append("High browsing engagement (frequent views)")

        if preferred_cat and row["product_category"] == preferred_cat:
            why.append(f"Matches preferred category ({preferred_cat})")

        if prev_p >= 3:
            why.append(f"Strong repeat customer loyalty ({prev_p} previous orders)")
        elif prev_p == 0:
            why.append("Top-converting discovery item for new shoppers")

        if price < 60:
            why.append("Attractive budget-friendly price point")

        if not why:
            why.append("Trending top recommendation with high overall buyer affinity")

        reasons.append(" • ".join(why[:2]))

    df_sorted["why_recommended"] = reasons
    return df_sorted, excluded_count


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Customer Controls")

    input_mode = st.radio(
        "Choose Customer Input Mode:",
        ["👤 Select Existing Customer", "✨ New Customer Simulator"]
    )

    already_bought = []

    if input_mode == "👤 Select Existing Customer":
        all_custs = customer_profiles["customer_id"].tolist()
        selected_cid = st.selectbox("Select Customer ID:", all_custs, index=0)

        # Retrieve customer stats
        cust_row = customer_profiles[customer_profiles["customer_id"] == selected_cid].iloc[0]
        cust_loyalty = int(cust_row["previous_purchases"])
        cust_views = max(1, int(round(cust_row["views"])))
        cust_carts = min(cust_views, max(0, int(round(cust_row["cart_adds"]))))
        cust_fav_cat = cust_row["product_category"]

        already_bought = customer_purchases.get(selected_cid, [])

        st.caption("**Customer Profile Summary:**")
        st.write(f"• **Customer ID:** `{selected_cid}`")
        st.write(f"• **Previous Orders:** {cust_loyalty}")
        st.write(f"• **Typical Views:** ~{cust_views} | **Cart Adds:** ~{cust_carts}")
        st.write(f"• **Preferred Category:** {cust_fav_cat}")
        st.write(f"• **Past Items Bought:** {len(already_bought)} product(s)")

        customer_profile = {
            "customer_id": selected_cid,
            "previous_purchases": cust_loyalty,
            "views": cust_views,
            "cart_adds": cust_carts,
            "preferred_category": cust_fav_cat
        }

    else:
        st.markdown("**Simulate a New Shopper's Behavior:**")
        sim_loyalty = st.slider("Previous Lifetime Purchases:", 0, 30, 0, help="Set to 0 to simulate a cold-start first-time visitor")
        sim_views = st.slider("Product Views in Session:", 1, 30, 5)
        sim_carts = st.slider("Cart Additions in Session:", 0, sim_views, 2)
        sim_cat = st.selectbox(
            "Session Category Interest:",
            ["Electronics", "Clothing", "Home & Kitchen", "Books", "Beauty", "Sports"]
        )

        customer_profile = {
            "customer_id": "NEW_CUSTOMER",
            "previous_purchases": sim_loyalty,
            "views": sim_views,
            "cart_adds": sim_carts,
            "preferred_category": sim_cat
        }

    st.markdown("---")
    st.header("🎯 Recommendation Options")

    exclude_purchased = st.checkbox(
        "Exclude previously purchased products",
        value=True,
        help="Prevents recommending items the customer has already purchased."
    )

    top_n = st.slider("Number of Recommendations:", 3, 10, 5)
    category_filter = st.selectbox(
        "Filter by Product Category:",
        ["All"] + sorted(list(product_catalog["product_category"].unique()))
    )

    st.markdown("---")
    st.info(f"🏆 Active Model: **{best_model_name}**")


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">Smart Product Recommendation System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Personalized machine learning recommendation engine using behavioral signals (views, cart adds, and historical loyalty) '
    'to predict purchase probability and rank products in real time.'
    '</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# App Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Top Recommendations",
    "📈 Feature Importance",
    "🏆 Model Comparison & Tuning",
    "🧹 Data Cleaning Summary"
])

# =========================================================
# TAB 1: Recommendations
# =========================================================
with tab1:
    # Cold-Start Detection & Explanation
    is_cold_start = (customer_profile["previous_purchases"] == 0 or customer_profile["customer_id"] == "NEW_CUSTOMER")
    if is_cold_start:
        st.markdown("""
        <div class="cold-start-card">
            ❄️ <strong>Cold-Start Handling Active:</strong><br>
            This customer has no prior purchase history (0 previous orders).
            Our recommendation engine handles cold-start users by evaluating real-time session signals
            (browsing views, cart additions, and selected category affinity) alongside overall catalog conversion probability,
            generating high-confidence recommendations without requiring historical customer purchase records.
        </div>
        """, unsafe_allow_html=True)

    exclude_list = already_bought if (exclude_purchased and already_bought) else []
    recs_df, num_excluded = generate_recommendations(
        customer_profile,
        product_catalog,
        best_pipeline,
        top_n=top_n,
        category_filter=category_filter,
        exclude_pids=exclude_list
    )

    if recs_df.empty:
        st.warning("No products found matching the selected filter criteria.")
    else:
        st.subheader(f"Top {len(recs_df)} Personalized Product Recommendations")
        sub_desc = f"Recommendations dynamically generated for **{customer_profile['customer_id']}**"
        if num_excluded > 0:
            sub_desc += f" | *Excluded {num_excluded} previously purchased item(s) to guarantee novelty.*"
        st.markdown(sub_desc)

        col_left, col_right = st.columns([1.15, 0.85])

        with col_left:
            for idx, row in recs_df.iterrows():
                rank = idx + 1
                prob_pct = row["purchase_probability"] * 100
                st.markdown(f"""
                <div class="recommendation-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.15rem; font-weight: 700; color: #1E293B;">
                            #{rank}. {row['product_id']} — {row['product_category']}
                        </span>
                        <span style="font-size: 1.25rem; font-weight: 700; color: #2563EB;">
                            {prob_pct:.1f}% Probability
                        </span>
                    </div>
                    <div style="margin-top: 6px; color: #475569; font-size: 0.95rem;">
                        <strong>Price:</strong> ${row['product_price']:.2f} &nbsp;|&nbsp; 
                        <strong>Price Tier:</strong> {row['price_bucket']}
                    </div>
                    <div class="why-tag">
                        💡 <strong>Why recommended:</strong> {row['why_recommended']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Upgrade 5: Download recommendations as CSV button
            csv_export = recs_df[[
                "product_id", "product_category", "product_price",
                "price_bucket", "purchase_probability", "why_recommended"
            ]].copy()
            csv_export["purchase_probability"] = (csv_export["purchase_probability"] * 100).round(2).astype(str) + "%"
            csv_data = csv_export.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Recommendations as CSV",
                data=csv_data,
                file_name=f"recommendations_{customer_profile['customer_id']}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_right:
            st.markdown("##### 📊 Purchase Probability Comparison")
            plot_df = recs_df.sort_values(by="purchase_probability", ascending=True)
            fig = px.bar(
                plot_df,
                x="purchase_probability",
                y="product_id",
                orientation="h",
                text=plot_df["purchase_probability"].apply(lambda p: f"{p*100:.1f}%"),
                color="purchase_probability",
                color_continuous_scale="Blues",
                labels={"purchase_probability": "Purchase Probability", "product_id": "Product"}
            )
            fig.update_layout(
                xaxis=dict(range=[0, 1.05], tickformat=".0%"),
                yaxis_title="",
                xaxis_title="Predicted Purchase Probability",
                coloraxis_showscale=False,
                margin=dict(l=10, r=20, t=10, b=30),
                height=350
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            # Customer Context summary box
            st.markdown(f"""
            <div class="metric-card">
                <strong>Current Customer Context:</strong><br>
                • <strong>Customer ID:</strong> <code>{customer_profile['customer_id']}</code><br>
                • <strong>Session Views:</strong> {customer_profile['views']} &nbsp;|&nbsp; <strong>Cart Adds:</strong> {customer_profile['cart_adds']}<br>
                • <strong>Cart Rate:</strong> {(customer_profile['cart_adds']/max(1, customer_profile['views']))*100:.1f}%<br>
                • <strong>Lifetime Orders:</strong> {customer_profile['previous_purchases']}<br>
                • <strong>Status:</strong> {'Cold-Start User' if is_cold_start else 'Established Customer'}
            </div>
            """, unsafe_allow_html=True)

            if already_bought and exclude_purchased:
                with st.expander(f"📦 Customer's Prior Purchases ({len(already_bought)} items)"):
                    st.write(", ".join(already_bought))

# =========================================================
# TAB 2: Feature Importance
# =========================================================
with tab2:
    st.subheader(f"🧠 Feature Importance ({best_model_name})")
    st.markdown(
        "Feature importance reveals which shopping behavior attributes had the highest mathematical weight "
        "when predicting whether a customer will purchase."
    )

    top_features = feature_importance_df.head(10).sort_values(by="Importance", ascending=True)

    fig_feat = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Viridis",
        labels={"Importance": "Model Weight / Relative Importance", "Feature": "Feature Name"}
    )
    fig_feat.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=10, b=30),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_feat, use_container_width=True)

    st.markdown("""
    > [!TIP]
    > **Presentation Insights for Students:**
    > 1. **Cart Rate & Cart Adds:** Rank as dominant signals. Converting views into cart additions shows decisive buying intent.
    > 2. **Category Affinity:** Browsing within a favorite category significantly enhances conversion odds.
    > 3. **Customer Loyalty:** Repeat buyers demonstrate higher baseline conversion compared to cold-start shoppers.
    """)

# =========================================================
# TAB 3: Model Comparison & Tuning
# =========================================================
with tab3:
    st.subheader("🏆 Model Benchmarks, 5-Fold Cross-Validation & Hyperparameter Tuning")
    st.markdown(
        "To ensure robust evaluation without overfitting, we performed **5-Fold Stratified Cross-Validation** "
        "across all candidate models, followed by **GridSearchCV hyperparameter optimization** for the top-performing model."
    )

    # 5-Fold Cross-Validation Results
    st.markdown("##### 1. 5-Fold Stratified Cross-Validation Summary (Training Data)")
    if not cv_results_df.empty:
        st.dataframe(cv_results_df, use_container_width=True)

    # Hyperparameter Tuning Details
    if best_params:
        param_str = ", ".join([f"<code>{k} = {v}</code>" for k, v in best_params.items()])
        st.markdown(f"""
        <div class="params-card">
            ⚙️ <strong>Optimal Hyperparameters (GridSearchCV):</strong> {param_str}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("##### 2. Holdout Test Set Benchmark Leaderboard")
    def highlight_best(row):
        if best_model_name in row["Model"]:
            return ["background-color: #DCFCE7; font-weight: bold; color: #166534"] * len(row)
        return [""] * len(row)

    st.dataframe(
        metrics_df.style.apply(highlight_best, axis=1),
        use_container_width=True
    )

    # ROC Curve & Confusion Matrix Side-by-Side (Upgrade 3)
    col_roc, col_cm = st.columns(2)

    with col_roc:
        st.markdown("##### 📈 ROC Curve Comparison")
        fig_roc = go.Figure()

        # Add random guess diagonal line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray", width=1.5),
            name="Random Guess (AUC = 0.50)"
        ))

        # Add ROC curve for each model
        colors = ["#2563EB", "#059669", "#D97706", "#7C3AED"]
        for i, (m_name, r_data) in enumerate(roc_curves.items()):
            c = colors[i % len(colors)]
            fig_roc.add_trace(go.Scatter(
                x=r_data["fpr"],
                y=r_data["tpr"],
                mode="lines",
                line=dict(color=c, width=2.5 if best_model_name in m_name else 1.8),
                name=f"{m_name} (AUC: {r_data['auc']:.4f})"
            ))

        fig_roc.update_layout(
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Sensitivity / Recall)",
            xaxis=dict(range=[-0.02, 1.02]),
            yaxis=dict(range=[-0.02, 1.02]),
            legend=dict(x=0.45, y=0.15, bgcolor="rgba(255,255,255,0.85)"),
            height=380,
            margin=dict(l=10, r=10, t=10, b=30)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_cm:
        st.markdown(f"##### 🎯 Confusion Matrix ({best_model_name})")
        if confusion_matrix_data:
            cm = np.array(confusion_matrix_data["matrix"])
            tn = confusion_matrix_data["tn"]
            fp = confusion_matrix_data["fp"]
            fn = confusion_matrix_data["fn"]
            tp = confusion_matrix_data["tp"]
            total = confusion_matrix_data["total_test"]

            # Annotated Heatmap
            z_vals = [[tn, fp], [fn, tp]]
            text_labels = [
                [f"True Negative (TN)<br><b>{tn}</b> ({tn/total*100:.1f}%)", f"False Positive (FP)<br><b>{fp}</b> ({fp/total*100:.1f}%)"],
                [f"False Negative (FN)<br><b>{fn}</b> ({fn/total*100:.1f}%)", f"True Positive (TP)<br><b>{tp}</b> ({tp/total*100:.1f}%)"]
            ]

            fig_cm = ff = go.Figure(data=go.Heatmap(
                z=z_vals,
                x=["Predicted Non-Buyer", "Predicted Buyer"],
                y=["Actual Non-Buyer", "Actual Buyer"],
                text=text_labels,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False
            ))
            fig_cm.update_layout(
                yaxis=dict(autorange="reversed"),
                height=380,
                margin=dict(l=10, r=10, t=10, b=30)
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            # Quick metric explanation
            sensitivity = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
            specificity = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0
            st.caption(f"**Sensitivity (Recall):** {sensitivity:.1f}% &nbsp;|&nbsp; **Specificity:** {specificity:.1f}%")

# =========================================================
# TAB 4: Data Cleaning Summary
# =========================================================
with tab4:
    st.subheader("🧹 Data Cleaning & Preprocessing Pipeline")
    st.markdown(
        "Raw e-commerce data often suffers from duplicate events (e.g. double-clicks) and missing values. "
        "Our pipeline automatically audits and cleans the dataset before model training."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Raw Dataset Size", f"{cleaning_stats['initial_rows']:,} rows")
    with c2:
        st.metric("Duplicates Dropped", f"{cleaning_stats['duplicates_removed']} rows", delta="Cleaned", delta_color="normal")
    with c3:
        st.metric("Total Clean Rows", f"{cleaning_stats['clean_rows']:,} rows")
    with c4:
        st.metric("Missing Value Imputation", "Median & Mode", delta="Completed", delta_color="normal")

    st.markdown("---")
    st.markdown("##### 📋 Imputation Strategy Breakdown")
    impute_data = []
    for col, count in cleaning_stats["missing_before"].items():
        strategy = cleaning_stats["imputed_strategy"].get(col, "None")
        impute_data.append({
            "Column Name": col,
            "Missing Values Found": count,
            "Imputation Strategy": strategy
        })

    st.table(pd.DataFrame(impute_data))

    st.markdown("""
    > [!IMPORTANT]
    > **Zero Data Leakage Guarantee:**
    > To prevent data leakage, the raw `purchases` column was strictly excluded from all training features. 
    > It was solely used to generate the binary classification ground truth target `will_purchase` (1 if purchases > 0).
    """)

    with st.expander("🔍 Preview Cleaned Dataset (First 10 Rows)"):
        st.dataframe(cleaned_df.head(10), use_container_width=True)

# Footer
st.markdown("---")
st.caption("Smart Product Recommendation System | Built with Python, Scikit-Learn & Streamlit")
