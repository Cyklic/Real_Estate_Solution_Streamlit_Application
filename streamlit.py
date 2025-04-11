import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st
import logging
import traceback

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="a"
)

logging.info("Real Estate Price Predictor app started.")

# Set the page title and description
st.markdown("<h1 style='text-align: center;'>Real Estate Price Predictor</h1>", unsafe_allow_html=True)

st.write("""
This app predicts the price of housing properties based on its characteristics.
""")

# # Optional password protection (remove if not needed)
# password_guess = st.text_input("Please enter your password?")
# # this password is stores in streamlit secrets
# if password_guess != st.secrets["password"]:
#     st.stop()

# Load the pre-trained model
try:
    with open("models/LRmodel.pkl", "rb") as lr_pickle:
        lr_model = pickle.load(lr_pickle)
    logging.info("Linear Regression model loaded successfully.")
except Exception as e:
    logging.error("Failed to load the model.")
    logging.error(traceback.format_exc())
    st.error("Error loading the prediction model. Please try again later.")
    st.stop()

# Prepare the form to collect user inputs
with st.form("user_inputs"):
    st.markdown("<h3 style='text-align: center;'>Property Details</h3>", unsafe_allow_html=True)
    
    # Create 5 columns
    cols = st.columns(5)

    # Row 1
    # Column 1
    with cols[0]:
        year_sold = st.number_input("Year Sold", min_value=0, step=100)
    # Column 2
    with cols[1]:
        property_tax = st.number_input("Property Tax", min_value=0, step=100)
    # Column 3
    with cols[2]:
        insurance = st.number_input("Insurance", min_value=0, step=100)
    # Column 4
    with cols[3]:
        beds = st.number_input("Beds Count", min_value=0, step=1)
    # Column 5
    with cols[4]:
        baths = st.number_input("Baths Count", min_value=0, step=1)

    # Row 2
    cols = st.columns(5)
    
    # Column 1
    with cols[0]:
        sqft = st.number_input("Square Feet", min_value=0, step=1000)
    # Column 2
    with cols[1]:
        year_built = st.number_input("Year Built", min_value=0, step=100)
    # Column 3
    with cols[2]:
        lot_size = st.number_input("Lot Size", min_value=0, step=1000)
    # Column 4
    with cols[3]:
        bsemnt = st.selectbox("Basement", options=["Yes", "No"])
    # Column 5
    with cols[4]:
        pop = st.selectbox("Popular", options=["Yes", "No"])

    # Row 3
    cols = st.columns(5)
    
    # Column 1
    with cols[0]:
        recess = st.selectbox("Recession", options=["Yes", "No"])
    # Column 2
    with cols[1]:
        property_age = st.number_input("Property Age", min_value=0, step=10)
    # Column 3
    with cols[2]:
        prop_type = st.selectbox("Property Type", options=["Bunglow", "Condo"])
    # Column 5
    with cols[4]:
        submitted = st.form_submit_button("Predict Property Price")

# Handle the dummy variables to pass to the model
if submitted:
    try:
        # Handle dependents
        basement = 1 if bsemnt == "Yes" else 0
        popular = 1 if pop == "Yes" else 0
        recession = 1 if recess == "Yes" else 0
        property_type_Bunglow = 1 if prop_type == "Bunglow" else 0
        property_type_Condo = 1 if prop_type == "Condo" else 0

        # Prepare the input for prediction. This has to go in the same order as it was trained
        feature_names = [
            'year_sold', 'property_tax', 'insurance', 'beds', 'baths', 'sqft',
            'year_built', 'lot_size', 'basement', 'popular', 'recession',
            'property_age', 'property_type_Bunglow', 'property_type_Condo'
        ]

        prediction_input = pd.DataFrame([[
            year_sold, property_tax, insurance, beds, baths, sqft,
            year_built, lot_size, basement, popular, recession,
            property_age, property_type_Bunglow, property_type_Condo
        ]], columns=feature_names)

        # Make prediction
        new_prediction = lr_model.predict(prediction_input)[0]
        logging.info(f"Prediction made successfully: {new_prediction}")

        # Display result
        st.subheader("Prediction Result:")
        st.success(f"Your Property Price Prediction is: {new_prediction}")

        try:
            st.image("combined_violin_plots.png")
        except Exception as e:
            logging.warning("Could not display violin plot image.")
            logging.warning(traceback.format_exc())

    except Exception as e:
        logging.error("Error occurred during prediction process.")
        logging.error(traceback.format_exc())
        st.error("An error occurred during prediction. Please check your inputs and try again.")

st.write(
    """We used a machine learning (Linear Regression) model to predict property price based on the above characteristics."""
)


