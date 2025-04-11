# Real Estate Price Predictor Application
This app has been built using Streamlit and deployed with Streamlit community cloud

[Visit the app here](https://leonard-umoru-real-estate-solution.streamlit.app/)

This application aims to predicts the price of housing properties based on inputs.

## Features
- User-friendly interface powered by Streamlit.
- Input form to enter details such as year sold, property tax, bed count, bath count, and other relevant factors.
- Real-time prediction of property prices based on the trained model.
- Accessible via Streamlit Community Cloud.

## Technologies Used
- **Streamlit**: For building the web application.
- **Scikit-learn**: For model training and evaluation.
- **Pandas** and **NumPy**: For data preprocessing and manipulation.
- **Matplotlib** and **Seaborn**: For exploratory data analysis and visualization (if applicable).

## Model
The predictive model applies preprocessing steps like encoding categorical variables and scaling numerical features. The regression model used may include algorithms such as Linear Regression, Decision Tree, and Random Forest.

## Future Enhancements
* Adding support for multiple datasets.
* Incorporating explainability tools like SHAP to provide insights into predictions.
* Adding visualizations to better represent user input and model predictions.

## Installation (for local deployment)
If you want to run the application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/credit_eligibility_application.git
   cd credit_eligibility_application

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\\Scripts\\activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the Streamlit application:
   ```bash
   streamlit run app.py

#### Thank you for using the Real Estate Price Predictor Application! Feel free to share your feedback.
