import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    categorical_columns = ['Home Ownership', 'Verification Status', 'Loan Title', 'Application Type']
    data_encoded = pd.get_dummies(data, columns=categorical_columns)
    return data, data_encoded  # Return both raw and encoded data

# Train the model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_resampled, y_train_resampled)

    return model, scaler, X_test_scaled, y_test

# Main Streamlit app
def main():
    st.title("Loan Status Prediction App")
    st.write("Upload your loan dataset and predict loan status (0 = Non-Default, 1 = Default)")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        raw_data, data_encoded = load_data(uploaded_file)
        X = data_encoded.drop('Loan Status', axis=1)
        y = data_encoded['Loan Status']

        # Train model
        model, scaler, X_test_scaled, y_test = train_model(X, y)

        # Predictions with adjustable threshold
        threshold = st.slider("Prediction Threshold", 0.1, 0.9, 0.3, 0.05)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Evaluation metrics
        st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
        st.write(f"**ROC-AUC:** {roc_auc_score(y_test, y_pred_proba):.2f}")
        st.write("**Classification Report:**")
        st.text(classification_report(y_test, y_pred))

        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        st.write("**Top 10 Most Important Features:**")
        st.dataframe(feature_importance.head(10))

        # Plot feature importance
        fig, ax = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), ax=ax)
        ax.set_title("Top 10 Feature Importance")
        st.pyplot(fig)

        # User input for prediction
        st.subheader("Predict Loan Status")
        input_data = {}
        for col in ['Total Current Balance', 'Loan Amount', 'Interest Rate', 'Annual Income', 
                    'Last week Pay', 'Debit to Income', 'Term']:  # Added 'Term'
            input_data[col] = st.number_input(col, value=float(raw_data[col].mean()))

        # Categorical inputs
        home_ownership = st.selectbox("Home Ownership", ['MORTGAGE', 'RENT', 'OWN'])
        verification_status = st.selectbox("Verification Status", ['Not Verified', 'Source Verified', 'Verified'])
        loan_title = st.selectbox("Loan Title", raw_data['Loan Title'].unique())
        application_type = st.selectbox("Application Type", ['INDIVIDUAL', 'JOINT'])

        if st.button("Predict"):
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            categorical_columns = ['Home Ownership', 'Verification Status', 'Loan Title', 'Application Type']
            for col, value in zip(categorical_columns, [home_ownership, verification_status, loan_title, application_type]):
                for unique_val in raw_data[col].unique():
                    input_df[f"{col}_{unique_val}"] = 1 if unique_val == value else 0
            input_scaled = scaler.transform(input_df[X.columns])
            pred_proba = model.predict_proba(input_scaled)[0, 1]
            pred = 1 if pred_proba >= threshold else 0
            st.write(f"Prediction Probability (Class 1): {pred_proba:.2f}")
            st.write(f"Predicted Loan Status: {'Default (1)' if pred == 1 else 'Non-Default (0)'}")

if __name__ == "__main__":
    main()