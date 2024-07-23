import pickle as pkl
import pandas as pd

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
    

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

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