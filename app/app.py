# ============================================
# AGRICULTURAL AI YIELD PREDICTION DASHBOARD
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt


# ============================================
# 1. PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="AI Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)


# ============================================
# 2. LOAD MODEL
# ============================================
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "final_yield_model.pkl"

model = joblib.load(MODEL_PATH)








# ============================================
# 3. LOAD FEATURE IMPORTANCE
# ============================================

importance_path = "feature_importance.csv"

if os.path.exists(importance_path):
    feature_importance = pd.read_csv(importance_path)
else:
    feature_importance = None


# ============================================
# 4. TITLE
# ============================================

st.title("🌾 AI-Based Crop Yield Prediction System")

st.markdown(
    """
    This system predicts agricultural crop yield using a machine-learning
    model trained on environmental, weather, agricultural management,
    crop and regional features.
    """
)

st.divider()


# ============================================
# 5. SIDEBAR
# ============================================

st.sidebar.header("System Information")

st.sidebar.write(
    "Model: HistGradientBoosting"
)

st.sidebar.write(
    "Prediction Target: yield_tons_per_ha"
)

st.sidebar.write(
    "Random Seed: 42"
)

st.sidebar.info(
    "Upload a CSV containing the same input features used during model training."
)


# ============================================
# 6. FILE UPLOAD
# ============================================

uploaded_file = st.file_uploader(
    "Upload agricultural CSV dataset",
    type=["csv"]
)


# ============================================
# 7. PROCESS DATA
# ============================================

if uploaded_file is not None:

    try:

        new_data = pd.read_csv(uploaded_file)

        st.success(
            f"Dataset loaded successfully: {new_data.shape[0]} rows, "
            f"{new_data.shape[1]} columns."
        )

        # ----------------------------------------
        # Display uploaded data
        # ----------------------------------------

        st.subheader("Uploaded Dataset")

        st.dataframe(
            new_data,
            use_container_width=True
        )


        # ========================================
        # REMOVE TARGET IF USER UPLOADS IT
        # ========================================

        target = "yield_tons_per_ha"

        prediction_data = new_data.copy()

        if target in prediction_data.columns:

            actual_values = prediction_data[target].copy()

            prediction_data = prediction_data.drop(
                columns=[target]
            )

        else:

            actual_values = None


        # ========================================
        # REMOVE ID COLUMNS
        # ========================================

        possible_id_columns = [
            "plot_id",
            "farm_id",
            "farmer_id",
            "household_id",
            "record_id",
            "id"
        ]

        id_columns = [
            col for col in possible_id_columns
            if col in prediction_data.columns
        ]

        prediction_data_model = prediction_data.drop(
            columns=id_columns,
            errors="ignore"
        )


        # ========================================
        # CHECK MODEL FEATURES
        # ========================================

        expected_features = model.named_steps[
            "preprocessor"
        ].feature_names_in_

        missing_features = [
            col for col in expected_features
            if col not in prediction_data_model.columns
        ]

        extra_features = [
            col for col in prediction_data_model.columns
            if col not in expected_features
        ]


        # ========================================
        # HANDLE MISSING FEATURES
        # ========================================

        if len(missing_features) > 0:

            st.error(
                "The uploaded dataset is missing required model features."
            )

            st.write("Missing features:")

            st.write(missing_features)

            st.stop()


        # ========================================
        # KEEP ONLY MODEL FEATURES
        # ========================================

        prediction_data_model = prediction_data_model[
            expected_features
        ]


        # ========================================
        # MAKE PREDICTIONS
        # ========================================

        predictions = model.predict(
            prediction_data_model
        )


        # ========================================
        # ADD PREDICTIONS
        # ========================================

        results = new_data.copy()

        results["predicted_yield_tons_per_ha"] = predictions


        # ========================================
        # DISPLAY RESULTS
        # ========================================

        st.subheader("Prediction Results")

        st.dataframe(
            results,
            use_container_width=True
        )


        # ========================================
        # SUMMARY METRICS
        # ========================================

        st.subheader("Prediction Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Number of Plots",
                len(predictions)
            )

        with col2:

            st.metric(
                "Mean Predicted Yield",
                f"{np.mean(predictions):.2f} tons/ha"
            )

        with col3:

            st.metric(
                "Minimum Predicted Yield",
                f"{np.min(predictions):.2f} tons/ha"
            )

        with col4:

            st.metric(
                "Maximum Predicted Yield",
                f"{np.max(predictions):.2f} tons/ha"
            )


        # ========================================
        # ACTUAL VS PREDICTED
        # ========================================

        if actual_values is not None:

            st.subheader("Actual vs Predicted Performance")

            from sklearn.metrics import (
                mean_absolute_error,
                mean_squared_error,
                r2_score
            )

            rmse = np.sqrt(
                mean_squared_error(
                    actual_values,
                    predictions
                )
            )

            mae = mean_absolute_error(
                actual_values,
                predictions
            )

            r2 = r2_score(
                actual_values,
                predictions
            )

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric(
                    "RMSE",
                    f"{rmse:.4f}"
                )

            with m2:
                st.metric(
                    "MAE",
                    f"{mae:.4f}"
                )

            with m3:
                st.metric(
                    "R²",
                    f"{r2:.4f}"
                )

            # ------------------------------------
            # Scatter plot
            # ------------------------------------

            fig, ax = plt.subplots(
                figsize=(8, 6)
            )

            ax.scatter(
                actual_values,
                predictions,
                alpha=0.6
            )

            min_value = min(
                actual_values.min(),
                predictions.min()
            )

            max_value = max(
                actual_values.max(),
                predictions.max()
            )

            ax.plot(
                [min_value, max_value],
                [min_value, max_value],
                linestyle="--"
            )

            ax.set_xlabel(
                "Actual Yield (tons/ha)"
            )

            ax.set_ylabel(
                "Predicted Yield (tons/ha)"
            )

            ax.set_title(
                "Actual vs Predicted Yield"
            )

            st.pyplot(fig)


        # ========================================
        # PREDICTION DISTRIBUTION
        # ========================================

        st.subheader("Predicted Yield Distribution")

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.hist(
            predictions,
            bins=30
        )

        ax.set_xlabel(
            "Predicted Yield (tons/ha)"
        )

        ax.set_ylabel(
            "Number of Plots"
        )

        ax.set_title(
            "Distribution of Predicted Crop Yield"
        )

        st.pyplot(fig)


        # ========================================
        # FEATURE IMPORTANCE
        # ========================================

        if feature_importance is not None:

            st.subheader(
                "Top Features Influencing Prediction"
            )

            top_features = feature_importance.head(15)

            fig, ax = plt.subplots(
                figsize=(10, 7)
            )

            plot_data = top_features.sort_values(
                "importance_mean"
            )

            ax.barh(
                plot_data["feature"],
                plot_data["importance_mean"]
            )

            ax.set_xlabel(
                "Permutation Importance"
            )

            ax.set_ylabel(
                "Feature"
            )

            ax.set_title(
                "Top 15 Model Features"
            )

            st.pyplot(fig)


        # ========================================
        # DOWNLOAD RESULTS
        # ========================================

        csv_data = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Predictions CSV",
            data=csv_data,
            file_name="crop_yield_predictions.csv",
            mime="text/csv"
        )


    except Exception as e:

        st.error(
            f"Prediction failed: {str(e)}"
        )


else:

    # ============================================
    # INITIAL SCREEN
    # ============================================

    st.info(
        "Upload a CSV dataset to generate crop-yield predictions."
    )

    st.write(
        "Expected prediction target:"
    )

    st.code(
        "yield_tons_per_ha"
    )
