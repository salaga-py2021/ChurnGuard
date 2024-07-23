import pickle as pkl
import pandas as pd
from openai import OpenAI
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

def predict_from_pkl(pkl_path,new_record=None,X_test=None, scaler=None):
    with open(pkl_path, 'rb') as f:
        # Load the model from file
        loaded_model = pkl.load(f)
        if new_record:
            # Sorts the record alphabetically and convert to a df
            new_record = dict(sorted(new_record.items()))
            new_record_df = pd.DataFrame([list(new_record.values())], columns=list(new_record.keys()))
            # Apply the Scaler used for the model
            if scaler:
                new_record_scaled = scaler.transform(new_record_df)
                new_record_df = pd.DataFrame(list(new_record_scaled)[0]).transpose()
                new_record_df.columns = list(new_record.keys())
            # Make a prediction
            prediction = loaded_model.predict(new_record_df)[0]
        else:
            y_pred = loaded_model.predict(X_test)
            return y_pred
    
    if prediction == 1:
        return True
    else:
        return False
    

# Function to detect and adjust outliers using IQR method
def adjust_outliers(df, columns):
    """Function to identify outliers using IQR method and adjust them for SpiderPlot"""
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

def create_fig(model_inputs):
    """Function to create spider plot"""
    df = pd.read_csv(Path("Cleaned_Modeling.csv"))
    df = df[['average_monthly_balance_prevQ','average_monthly_balance_prevQ2','branch_code','current_balance',
             'current_month_balance','current_month_debit','days_since_last_transaction','previous_month_balance',
             'previous_month_debit','previous_month_end_balance']]
    
    # Adjust outliers in the dataset
    columns_to_adjust = df.columns[1:-1]
    df_adjusted = adjust_outliers(df.copy(), columns_to_adjust)

    # Normalize the data for visualization purposes
    df_normalized = df_adjusted.copy()
    for column in df_normalized.columns[1:]:
        df_normalized[column] = df_normalized[column] / df_normalized[column].max()

    # Normalize the new customer data
    model_inputs_normalized = {k: v / df_adjusted[k].max() for k, v in model_inputs.items()}

    # Create the spider plot
    categories = list(df_normalized.columns[1:-1])
    values = list(model_inputs_normalized.values())

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='New Customer'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

def LLM(values,open_api_key,churn_prediction):
    # Prepare the prompt
    prompt = f"""You are a bank manager inspecting if a customer will churn/not churn in the near future. You have a data science model that is making predictions.
                Analyze the statistical measures of the variables from the training data and compare it with the variable values for the customer and create bullet points for the manager to understand why this prediction was made.
                Below are the statistical measures of the features from training data:
                1. Variable : current_balance, Mean : 7552.9258603283115 , Median : 3325.03, 25th Percentile : 1767.29, 75th Percentile : 6810.205
                2. Variable : current_month_debit, Mean : 4076.7408327040025 , Median : 182.3, 25th Percentile : 0.46, 75th Percentile : 1526.4099999999999
                3. Variable : previous_month_debit, Mean : 3725.2528192694467 , Median : 194.86, 25th Percentile : 0.47, 75th Percentile : 1557.2150000000001
                4. Variable : current_month_balance, Mean : 7624.336430700743 , Median : 3503.78, 25th Percentile : 2010.0149999999999, 75th Percentile : 6864.77
                5. Variable : average_monthly_balance_prevQ, Mean : 7660.709340593824 , Median : 3601.46, 25th Percentile : 2198.7200000000003, 75th Percentile : 6821.84
                6. Variable : previous_month_balance, Mean : 7654.748472912277 , Median : 3514.47, 25th Percentile : 2081.8199999999997, 75th Percentile : 6787.275
                7. Variable : previous_month_end_balance, Mean : 7661.40589252355 , Median : 3419.71, 25th Percentile : 1898.3049999999998, 75th Percentile : 6828.525
                8. Variable : average_monthly_balance_prevQ2, Mean : 7222.216939465004 , Median : 3368.14, 25th Percentile : 1797.945, 75th Percentile : 6617.585
                9. Variable : days_since_last_transaction, Mean : 167.17202591517946 , Median : 127.0, 25th Percentile : 108.0, 75th Percentile : 192.0
                10. Variable : previous_month_credit, Mean : 3679.4894650025835 , Median : 0.98, 25th Percentile : 0.36, 75th Percentile : 1048.52

                Below are the variable values for the customer under inspection :
                1. Variable : current_balance, Value : {values['current_balance']}
                2. Variable : current_month_debit, Value : {values['current_month_debit']}
                3. Variable : previous_month_debit, Value : {values['previous_month_debit']}
                4. Variable : current_month_balance, Value : {values['current_month_balance']}
                5. Variable : average_monthly_balance_prevQ, Value : {values['average_monthly_balance_prevQ']}
                6. Variable : previous_month_balance, Value : {values['previous_month_balance']}
                7. Variable : previous_month_end_balance, Value : {values['previous_month_end_balance']}
                8. Variable : average_monthly_balance_prevQ2, Value : {values['average_monthly_balance_prevQ2']}
                9. Variable : days_since_last_transaction, Value : {values['days_since_last_transaction']}
                10. Variable : branch_code, Value : {values['branch_code']}

                The predicted value for this customer is : {churn_prediction}
                0 means the customer will not churn. 1 means the customer will churn
                Your task : Provide precise bullet points which will help the manager understand the reasons for why the customer will churn or not by comparing the customers values for the variables to the population statistics. Do not include \n or \t. The output should directly be presentable as print. Please do not leave empty responses and only give 5 points"""
    # Initialize the OpenAI client with the API key
    client = OpenAI(api_key=open_api_key)

    # Use the chat completion endpoint
    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # Replace with the desired model
        messages=[
            {"role": "system", "content": "You are a helpful bank manager consultant named Tom."},
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content
