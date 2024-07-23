# Import libraries
import streamlit as st
import pandas as pd
import joblib
import os
import pandas as pd
from pathlib import Path
from helper_functions import predict_from_pkl, create_fig, LLM


# Set Paths
model_results_path = Path("./model_results")
pkl_path = model_results_path / Path("pkl")

# Page configuration
st.set_page_config(
    page_title="ChurnGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded")

# Custom CSS to remove margins from h1 and h2 elements
css = """
<style>
    h1, h2 {
        margin: 0 !important;
        padding: 0 !important;
    }
    p {
        padding: 0 !important;
        margin: 0 !important;
    }
</style>
"""

# Inject the CSS into the app
st.markdown(css, unsafe_allow_html=True)

# Title and Subtitle
# Title the app
st.title(':orange[ChurnGuard]')
st.subheader(':violet[Locking in Loyalty]')

# Logo images
st.sidebar.image('images/churnguard_transparent.png', width=150)
 
st.sidebar.markdown("## Select Model")
#Set model to be used
select_model = st.sidebar.selectbox('Which of the follwing models would you like to use?',
                                    ['Average Model', 'Random Forest', 'XGBoost', 'ADA Boost', 'Logistic Regression'])

#Choose default test_values
select_test_data = st.sidebar.selectbox('Which of the follwing test cases would you like to use?',
                                    ['Custom', 'Minimum', '25% Quartile', '50% Quartile', 'Average', '75% Quartile', 'Maximum', 
                                     'True Positive', 'True Negative', 'False Positive', 'False Negative'])


if select_test_data == "Custom":
    # Create a dictionary to store the data
    default_values = {
        'current_balance':None,
        'current_month_debit':None,
        'previous_month_debit':None,
        'days_since_last_transaction':None,
        'average_monthly_balance_prevQ':None,
        'current_month_balance':None,
        'previous_month_balance':None,
        'average_monthly_balance_prevQ2':None,
        'previous_month_end_balance':None,
        'branch_code':None
    }
else:
    # import the features
    feature_summary = pd.read_excel(model_results_path / Path("feature_summary.xlsx"), index_col=0)
    test_choice_map = {'Minimum':'min', '25% Quartile':'25%', '50% Quartile':'50%', 'Average': 'mean', '75% Quartile':'75%', 
                        'Maximum':'max', 'True Positive':'TP', 'True Negative':'TN', 'False Positive':'FP', 'False Negative':'FN'}
    default_values = {
        'current_balance':feature_summary.loc[test_choice_map[select_test_data]]['current_balance'],
        'current_month_debit': feature_summary.loc[test_choice_map[select_test_data]]['current_month_debit'],
        'previous_month_debit':feature_summary.loc[test_choice_map[select_test_data]]['previous_month_debit'],
        'days_since_last_transaction':feature_summary.loc[test_choice_map[select_test_data]]['days_since_last_transaction'],
        'average_monthly_balance_prevQ': feature_summary.loc[test_choice_map[select_test_data]]['average_monthly_balance_prevQ'],
        'current_month_balance':feature_summary.loc[test_choice_map[select_test_data]]['current_month_balance'],
        'previous_month_balance':feature_summary.loc[test_choice_map[select_test_data]]['previous_month_balance'],
        'average_monthly_balance_prevQ2':feature_summary.loc[test_choice_map[select_test_data]]['average_monthly_balance_prevQ2'],
        'previous_month_end_balance':feature_summary.loc[test_choice_map[select_test_data]]['previous_month_end_balance'],
        'branch_code':feature_summary.loc[test_choice_map[select_test_data]]['branch_code']
    }



st.sidebar.markdown("## Input Customer Data")
current_balance = st.sidebar.number_input("Insert customer's current balance", value=default_values['current_balance'], placeholder="Type a number...")
branch_code = st.sidebar.number_input("Insert customer's branch code", value=default_values['branch_code'], placeholder="Type a number...")
current_month_debit = st.sidebar.number_input("Insert customer's current month debit", value=default_values['current_month_debit'], placeholder="Type a number...")
previous_month_debit = st.sidebar.number_input("Insert customer's previous month debit", value=default_values['previous_month_debit'], placeholder="Type a number...")
current_month_balance = st.sidebar.number_input("Insert customer's current month balance", value=default_values['current_month_balance'], placeholder="Type a number...")
previous_month_balance = st.sidebar.number_input("Insert customer's previous month balance", value=default_values['previous_month_balance'], placeholder="Type a number...")
previous_month_end_balance = st.sidebar.number_input("Insert customer's previous month end balance", value=default_values['previous_month_end_balance'], placeholder="Type a number...")
average_monthly_balance_prevQ = st.sidebar.number_input("Insert customer's average monthly balance from the last quarter", value=default_values['average_monthly_balance_prevQ'], placeholder="Type a number...")
average_monthly_balance_prevQ2 = st.sidebar.number_input("Insert customer's average monthly balance from two quarters ago", value=default_values['average_monthly_balance_prevQ2'], placeholder="Type a number...")
days_since_last_transaction = st.sidebar.number_input("Insert customer's days since last transaction", value=default_values['days_since_last_transaction'], placeholder="Type a number...")


# Create a dictionary to store the data
model_inputs = {
    'current_balance':current_balance,
    'current_month_debit':current_month_debit,
    'previous_month_debit':previous_month_debit,
    'days_since_last_transaction':days_since_last_transaction,
    'average_monthly_balance_prevQ':average_monthly_balance_prevQ,
    'current_month_balance':current_month_balance,
    'previous_month_balance':previous_month_balance,
    'average_monthly_balance_prevQ2':average_monthly_balance_prevQ2,
    'previous_month_end_balance':previous_month_end_balance,
    'branch_code':branch_code
}

# Load the scaler
scaler = joblib.load(model_results_path / Path("MinMaxScaler.save"))

# load the model pkl file
model_path_mapper = {
    "Random Forest": pkl_path/"random_forest.pkl", 
    "XGBoost": pkl_path/"xg_model.pkl", 
    "ADA Boost": pkl_path/"ada_boost.pkl",
    "Logistic Regression": pkl_path/"logit.pkl"
}

if None not in model_inputs.values():
    if select_model != "Average Model":
        prediction = predict_from_pkl(
            pkl_path=model_path_mapper[select_model],
            new_record=model_inputs,
            X_test=None, 
            scaler=scaler
        )
    else:
        all_predictions = {}
        for model_name, model in model_path_mapper.items():
            all_predictions[model_name] = predict_from_pkl(
                pkl_path=model,
                new_record=model_inputs,
                X_test=None, 
                scaler=scaler
            )
        prediction_sum = sum(all_predictions.values())
        if prediction_sum > 2:
            prediction = True
        elif prediction_sum == 2:
            prediction = all_predictions["Random Forest"]
        else:
            prediction = False

    st.markdown("#### Model Prediction")
    if prediction == True:
        st.write("The customer is at risk of leaving the bank. It is recommended to take action to retain the customer. ")
        st.write("")
        prediction_image_col, prediction_text_col, empty_col = st.columns(3, vertical_alignment="center")
        with prediction_image_col:
            st.image('images/exit.png', width=200)
        with prediction_text_col:
            st.write("The customer is")
            st.header(":red[AT RISK]")
            st.write("of leaving the bank")
            
    else:
        st.write("The customer is not at risk of leaving the bank. No action is required at this time.")
        st.write("")
        prediction_image_col, prediction_text_col, empty_col = st.columns(3, vertical_alignment="center")
        with prediction_image_col:
            st.image('images/cyber-security.png', width=200)
        with prediction_text_col:
            st.write("The customer is")
            st.header(":green[NOT AT RISK]")
            st.write("of leaving the bank")
    
    st.markdown("#### Customer in Context")
    create_fig(model_inputs)

    st.markdown("#### Advice from ChatGPT")
    open_api_key = os.environ["OPENAI_API_KEY"]
    st.write(LLM(model_inputs,open_api_key,int(prediction)))


else:
    st.markdown("#### Model Prediction")
    st.write("Please input all the required data to make a prediction.")