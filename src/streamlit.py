import streamlit as st
import requests
import joblib

# Add some information about the service
st.title("Customer Deposit Prediction")
st.subheader("Just enter variabel below then click Predict button :sunglasses:")

# create form of input
with st.form(key = "air_data_form"):
    # create box number of input
    age = st.number_input(
        label = "1.\tEnter Age Value:",
        min_value = 18,
        max_value = 95,
        help = "Value range from 18 to 95"
    )

    balance = st.number_input(
        label = "2.\tEnter Balance Value[USD$]:",
        min_value = -1,
        max_value = 81204,
        help = "Value range from 0 to 81204"
    )

    duration = st.number_input(
        label = "3.\tEnter Duration Value (days):",
        min_value = 2,
        max_value = 3881,
        help = "Value range from 2 to 3881"
    )

    campaign = st.number_input(
        label ="4.\tEnter Campaign (Number of contact performed):",
        min_value = 1,
        max_value = 63,
        help = "Value range between 1 to 63"
    )

    pdays = st.number_input(
        label = "5.\tEnter Pdays (number of days that passed by after the client was last contacted):",
        min_value = -1,
        max_value = 854,
        help = "Value range between 0 to 854"
    )

    previous = st.number_input(
        label = "6.t\Enter Previous (number of contacts performed before this campaign):",
        min_value = 0,
        max_value = 58,
        help = "Value range between 0 to 58"
    )

    marital = st.selectbox(
        "7.\tEnter Marital Status:",
        ("married", "single", "divorced")
    )

    education = st.selectbox(
        "8.\tEnter Level of Education:",
        ("secondary", "Tertiary", "Primary", "Unknown")
    )

    default = st.selectbox(
        "9.\tEnter Default Status (Credit in defaul):",
        ("yes", "no")
    )

    housing = st.selectbox(
        "10.\Enter Housing Status (Housing Loan):",
        ("yes", "no")
    )

    loan = st.selectbox(
        "11.\tEnter Loan Status (Personal Loan):",
        ("yes", "no")
    )

    contact = st.selectbox(
        "12.\tEnter Contact (Type of communication):",
        ("unknown", "cellular", "telephone")
    )

    poutcome = st.selectbox(
        "13.\tEnter Poutcome (outcome of the previous marketing campaign):",
        ("unknown", "other", "failure", "success")
    )
    
    # Create submit buttin to the form
    submitted = st.form_submit_button("Predict")

    # Condition when form was submitted
    if submitted:
        # create dict of all data in the form
        raw_data = {
            "age" : age,
            "balance" : balance,
            "duration" : duration,
            "campaign" : campaign,
            "pdyas" : pdays,
            "previous" : previous,
            "marital" : marital,
            "education" : education,
            "default" : default,
            "housing" : housing,
            "loan" : loan,
            "contact" : contact,
            "poutcome" : poutcome
        }

        # Create loading animation while predicting
        with st.spinner("Predicting on process ..."):
            res = requests.post("http://api_backend:8080/predict", 
                                json = raw_data).json()
            
            # Parse the prediction result
            if res["error_msg"] != "":
                st.error("Error Occurs While Predicting: {}").format(res['error_msg'])
            else:
                if res['res'] != "yes":
                    st.warning("Predicted will not deposit")
                else:
                    st.success("Predicted will deposit")