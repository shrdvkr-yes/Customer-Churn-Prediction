import pandas as pd
import pickle as pkl
import streamlit as st


# ----------------------------
# PAGE CONFIG (must be first Streamlit command)
# ----------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)


# ----------------------------
# LOAD EXTERNAL CSS
# ----------------------------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ----------------------------
# LOAD MODEL & FEATURE COLUMNS
# ----------------------------
with open("model.pkl", "rb") as file:
    model = pkl.load(file)

with open("feature_columns.pkl", "rb") as file:
    feature_columns = pkl.load(file)


# ----------------------------
# TITLE (styled via .hero in style.css)
# ----------------------------
st.markdown("""
<div class="hero">
    <h1>📉 Customer Churn Prediction</h1>
    <p>Enter customer details to predict whether the customer may leave the service.</p>
</div>
""", unsafe_allow_html=True)


# ----------------------------
# INPUT FIELDS (3 columns)
# ----------------------------
st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    partner = st.selectbox("Partner", ["No", "Yes"])
with c2:
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
with c3:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=150, value=12)

st.markdown('<div class="section-title">Services & Contract</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
with c5:
    internet_service = st.selectbox("Internet Service", ["Dsl", "Fiber", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
with c6:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check",
        ],
    )

st.markdown('<div class="section-title">Charges</div>', unsafe_allow_html=True)
c7, c8 = st.columns(2)
with c7:
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
with c8:
    total_charges = st.number_input("Total Charges", min_value=0.0, value=840.0)


# ----------------------------
# ENCODING (must match what you used in training)
# ----------------------------
gender_encoded = {"Female": 0, "Male": 1}
yes_no_encoded = {"No": 0, "Yes": 1}
internet_encoded = {"Dsl": 0, "Fiber": 1, "No": 2}
contract_encoded = {"Month-to-month": 0, "One year": 1, "Two year": 2}
payment_encoded = {
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 1,
    "Electronic check": 2,
    "Mailed check": 3,
}


# ----------------------------
# PREDICTION
# ----------------------------
st.write("")
if st.button("Predict Churn"):

    customer_data = {
        "gender": gender_encoded[gender],
        "SeniorCitizen": senior_citizen,
        "Partner": yes_no_encoded[partner],
        "Dependents": yes_no_encoded[dependents],
        "tenure": tenure,
        "PhoneService": yes_no_encoded[phone_service],
        "InternetService": internet_encoded[internet_service],
        "Contract": contract_encoded[contract],
        "PaperlessBilling": yes_no_encoded[paperless_billing],
        "PaymentMethod": payment_encoded[payment_method],
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    # Build the DataFrame and put columns in the same order as training
    input_df = pd.DataFrame([customer_data])[feature_columns]

    prediction = model.predict(input_df)

    # Optional: show probability if the model supports it
    prob_text = ""
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_df)[0][1]
        prob_text = f"Churn probability: {prob:.1%}"

    if prediction[0] == 1:
        st.markdown(f"""
        <div class="result-card churn">
            <p>Prediction Result</p>
            <h2>⚠️ Likely to Churn</h2>
            <p>{prob_text}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card stay">
            <p>Prediction Result</p>
            <h2>✅ Likely to Stay</h2>
            <p>{prob_text}</p>
        </div>
        """, unsafe_allow_html=True)


# ----------------------------
# FOOTER
# ----------------------------
st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit &nbsp;|&nbsp; Customer Churn Model
</div>
""", unsafe_allow_html=True)