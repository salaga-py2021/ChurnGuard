#######################
# Import libraries
import random
from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt

#from langchain.chains import LLMChain
#from langchain_core.prompts import PromptTemplate
#from langchain_openai import OpenAI
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
from PIL import Image
from render_graphs import plotting
#from model import make_prediction
#from gpt import LLM
from typing import Iterator
from pathlib import Path
#######################
# Page configuration
st.set_page_config(
    page_title="US Population Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

# Load the image
image = Image.open(r'C:\Users\vkotr\New folder (2)\ChurnGuard\churnguard.jpg')

# Convert the image to RGBA format
image = image.convert("RGBA")

# Get the dimensions of the image
width, height = image.size

# Loop over each pixel in the image and set all white (also shades of whites)
# pixels to be transparent
for y in range(height):
    for x in range(width):
        pixel = image.getpixel((x, y))
        if pixel[0] > 200 and pixel[1] > 200 and pixel[2] > 200:
            image.putpixel((x, y), (255, 255, 255, 0))

image.save(
    r'C:\Users\vkotr\New folder (2)\ChurnGuard\churnguard_transparent.png', "PNG")

# Display the image
# st.image(image, width=200)

# Center the title

st.markdown("""
    <style>
        .big-font {
            font-size:50px !important;
            background: -webkit-linear-gradient(45deg, orange, yellow);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .small-font {
            font-size:30px !important;
            background: -webkit-linear-gradient(45deg, purple, blue);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
    """, unsafe_allow_html=True)


# Create three columns
col1, col2, col3 = st.columns([1, 6, 1])

# Display the image in the first column
col1.image(image, width=200)

# Display the title and subheader in the second column
col2.markdown('<p class="big-font" style="text-align: center; font-size: 30px;">Churn Guard</p>',
              unsafe_allow_html=True)
col2.markdown('<p class="small-font" style="text-align: center; font-size: 20px;">Locking in Loyalty</p>',
              unsafe_allow_html=True)


alt.themes.enable("dark")


#######################
# Load data
df_reshaped = pd.read_csv('data/us-population-2010-2019-reshaped.csv')

# selected_year = 2017
# df_selected_year = 2017
# df_selected_year_sorted = df_reshaped[df_reshaped.year == df_selected_year].sort_values(by="population", ascending=False)
# color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
# selected_color_theme = st.selectbox('Select a color theme', color_theme_list)

#######################
# Sidebar
selected_year = 2018
selected_color_theme = 'magma'
with st.sidebar:
    st.title('ChurnGuard')

    year_list = list(df_reshaped.year.unique())[::-1]

    selected_year = st.selectbox('Select a year', year_list)
    df_selected_year = df_reshaped[df_reshaped.year == selected_year]
    df_selected_year_sorted = df_selected_year.sort_values(
        by="population", ascending=False)

    color_theme_list = ['blues', 'cividis', 'greens', 'inferno',
                        'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    selected_color_theme = st.selectbox(
        'Select a color theme', color_theme_list)


#######################
# Plots


# Heatmap
def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme):
    heatmap = alt.Chart(input_df).mark_rect().encode(
        y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Year", titleFontSize=18,
                titlePadding=15, titleFontWeight=900, labelAngle=0)),
        x=alt.X(f'{input_x}:O', axis=alt.Axis(
            title="", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
        color=alt.Color(f'max({input_color}):Q',
                        legend=None,
                        scale=alt.Scale(scheme=input_color_theme)),
        stroke=alt.value('black'),
        strokeWidth=alt.value(0.25),
    ).properties(width=900
                 ).configure_axis(
        labelFontSize=12,
        titleFontSize=12
    )
    # height=300
    return heatmap

# Choropleth map


def make_choropleth(input_df, input_id, input_column, input_color_theme):
    choropleth = px.choropleth(input_df, locations=input_id, color=input_column, locationmode="USA-states",
                               color_continuous_scale=input_color_theme,
                               range_color=(input_df[input_column].min(
                               ), input_df[input_column].max()),
                               scope="usa",
                               labels={'population': 'Churn Rate'}
                               )
    choropleth.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=350
    )
    return choropleth


# Donut chart
def make_donut(input_response, input_text, input_color):
    if input_color == 'blue':
        chart_color = ['#29b5e8', '#155F7A']
    if input_color == 'green':
        chart_color = ['#27AE60', '#12783D']
    if input_color == 'orange':
        chart_color = ['#F39C12', '#875A12']
    if input_color == 'red':
        chart_color = ['#E74C3C', '#781F16']

    source = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100-input_response, input_response]
    })
    source_bg = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100, 0]
    })

    plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
        theta="% value",
        color=alt.Color("Topic:N",
                        scale=alt.Scale(
                            # domain=['A', 'B'],
                            domain=[input_text, ''],
                            # range=['#29b5e8', '#155F7A']),  # 31333F
                            range=chart_color),
                        legend=None),
    ).properties(width=130, height=130)

    text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=32,
                          fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
        theta="% value",
        color=alt.Color("Topic:N",
                        scale=alt.Scale(
                            # domain=['A', 'B'],
                            domain=[input_text, ''],
                            range=chart_color),  # 31333F
                        legend=None),
    ).properties(width=130, height=130)
    return plot_bg + plot + text

# Convert population to text


def format_number(num):
    if num > 1000000:
        if not num % 1000000:
            return f'{num // 1000000} M'
        return f'{round(num / 1000000, 1)} M'
    return f'{num // 1000} K'

# Calculation year-over-year population migrations


def calculate_population_difference(input_df, input_year):
    selected_year_data = input_df[input_df['year'] == input_year].reset_index()
    previous_year_data = input_df[input_df['year']
                                  == input_year - 1].reset_index()
    selected_year_data['population_difference'] = selected_year_data.population.sub(
        previous_year_data.population, fill_value=0)
    return pd.concat([selected_year_data.states, selected_year_data.id, selected_year_data.population, selected_year_data.population_difference], axis=1).sort_values(by="population_difference", ascending=False)

# '''
# customer_id,vintage,age,dependents,current_balance,previous_month_end_balance,average_monthly_balance_prevQ,average_monthly_balance_prevQ2,current_month_credit,previous_month_credit,current_month_debit,previous_month_debit,current_month_balance,previous_month_balance,churn,last_transaction,gender_Male,gender_other,occupation_retired,occupation_salaried,occupation_self_employed,occupation_student,customer_nw_category_2,customer_nw_category_3,days_since_last_transaction


# '''

global chat_value


def write(value):
    global chat_value
    chat_value = value


#######################
# Dashboard Main Panel
col = st.columns((1.5, 4.5, 2), gap='medium')

figure_value = 0

with col[0]:

    with st.form("my_form"):
        st.write("Customer Information Form")
        # slider_val = st.slider("Form slider")
        # checkbox_val = st.checkbox("Form checkbox")

        vintage = st.number_input("Vintage", step=1)
        age = st.number_input("Age", min_value=10, step=1)
        dependents = st.slider(
            "Dependents", min_value=0, max_value=10, step=1)
        current_balance = st.number_input("Current Balance", step=1)
        previous_month_end_balance = st.number_input(
            "Previous month end balance", step=1)
        average_monthly_balance_prevQ = st.number_input(
            "Average monthly balance prevQ", step=1)
        average_monthly_balance_prevQ2 = st.number_input(
            "Average monthly balance prevQ2", step=1)
        current_month_credit = st.number_input("Current month credit", step=1)
        previous_month_credit = st.number_input(
            "Previous month credit", step=1)
        current_month_debit = st.number_input("Current month debit", step=1)
        previous_month_debit = st.number_input("Previous month debit", step=1)
        current_month_balance = st.number_input(
            "Current month balance", step=1)
        previous_month_balance = st.number_input(
            "Previous month balance", step=1)
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
        occupation = st.selectbox("Occupation", options=[
                                  "Retired", "Salaried", "Self Employed", "Student"])
        customer_Networth = st.selectbox("Customer Networth", options=[
                                         'High', 'Medium', 'Low'])
        days_since_last_transaction = st.number_input(
            "Days since last transaction", step=1)
        branch_code = st.number_input("Branch Code", step=1)
        # loan_amount = st.number_input("loan amount", min_value=10000, max_value=100000, step=1)

        # Store inputs in a dictionary
        inputs = {
            "vintage": vintage,
            "age": age,
            "dependents": dependents,
            "current_balance": current_balance,
            "previous_month_end_balance": previous_month_end_balance,
            "average_monthly_balance_prevQ": average_monthly_balance_prevQ,
            "average_monthly_balance_prevQ2": average_monthly_balance_prevQ2,
            "current_month_credit": current_month_credit,
            "previous_month_credit": previous_month_credit,
            "current_month_debit": current_month_debit,
            "previous_month_debit": previous_month_debit,
            "current_month_balance": current_month_balance,
            "previous_month_balance": previous_month_balance,
            "gender": gender,
            "occupation": occupation,
            "customer_Networth": customer_Networth,
            "days_since_last_transaction": days_since_last_transaction,
            "branch_code": branch_code
        }

        variable_labels = {
            'current_balance': 'Current Balance',
            'current_month_debit': 'Current Month Debit',
            'previous_month_debit': 'Previous Month Debit',
            'current_month_balance': 'Current Month Balance',
            'average_monthly_balance_prevQ': 'Average Monthly Balance (Prev Q)',
            'previous_month_balance': 'Previous Month Balance',
            'previous_month_end_balance': 'Previous Month End Balance',
            'average_monthly_balance_prevQ2': 'Average Monthly Balance (Prev Q2)',
            'days_since_last_transaction': 'Days Since Last Transaction',
            'previous_month_credit': 'Previous Month Credit',
            # "branch_code": "Branch Code"
        }

        filtered_inputs = {key: inputs[key]
                           for key in variable_labels.keys() if key in inputs}
        # fig_value = plotting(filtered_inputs)

        print(filtered_inputs)

        # customer_churn = make_prediction(filtered_inputs, 'ada_model.pkl')

        # print(customer_churn)

        variable_labels_2 = {
            'current_balance': 'Current Balance',
            'current_month_debit': 'Current Month Debit',
            'previous_month_debit': 'Previous Month Debit',
            'current_month_balance': 'Current Month Balance',
            'average_monthly_balance_prevQ': 'Average Monthly Balance (Prev Q)',
            'previous_month_balance': 'Previous Month Balance',
            'previous_month_end_balance': 'Previous Month End Balance',
            'average_monthly_balance_prevQ2': 'Average Monthly Balance (Prev Q2)',
            'days_since_last_transaction': 'Days Since Last Transaction',
            'previous_month_credit': 'Previous Month Credit',
            "branch_code": "Branch Code",
            "hi": "hi"
        }

        filtered_inputs2 = {key: inputs[key]
                            for key in variable_labels_2.keys() if key in inputs}

        print(filtered_inputs2)

        # Convert dictionary to list
        inputs_list = list(inputs.values())

        print(inputs_list)

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write("Form submitted")
            figure_value = plotting(filtered_inputs)
            figure_value = 1
            # st.write("slider", , "checkbox", checkbox_val, "name", name, "credit score", credit_score, "loan amount", loan_amount)
            print("Form submitted")
            customer_churn = make_prediction(filtered_inputs2, 'rf.pkl')
            print(customer_churn)
            variable_labels_3 = {
                'current_balance': 'Current Balance',
                'current_month_debit': 'Current Month Debit',
                'previous_month_debit': 'Previous Month Debit',
                'current_month_balance': 'Current Month Balance',
                'average_monthly_balance_prevQ': 'Average Monthly Balance (Prev Q)',
                'previous_month_balance': 'Previous Month Balance',
                'previous_month_end_balance': 'Previous Month End Balance',
                'average_monthly_balance_prevQ2': 'Average Monthly Balance (Prev Q2)',
                'days_since_last_transaction': 'Days Since Last Transaction',
                'previous_month_credit': 'Previous Month Credit',
                'churn_prediction': customer_churn,
            }

            filtered_inputs3 = {
                key: inputs[key] for key in variable_labels_3.keys() if key in inputs}
            filtered_inputs3['churn_prediction'] = customer_churn

            #chat_value = LLM(filtered_inputs3)
            chat_value = "Temporarily Unavialable"

            print(chat_value)

            # Save chat_value to a text file
            with open('chat_value.txt', 'w') as f:
                f.write(str(chat_value))
                f.close()

            # write(chat_value)

        # st.write("Outside the form")

    # st.markdown('#### Gains/Losses')

    # df_population_difference_sorted = calculate_population_difference(df_reshaped, selected_year)

    # if selected_year > 2010:
    #     first_state_name = df_population_difference_sorted.states.iloc[0]
    #     first_state_population = format_number(df_population_difference_sorted.population.iloc[0])
    #     first_state_delta = format_number(df_population_difference_sorted.population_difference.iloc[0])
    # else:
    #     first_state_name = '-'
    #     first_state_population = '-'
    #     first_state_delta = ''
    # st.metric(label='india', value=first_state_population, delta=first_state_delta)

    # if selected_year > 2010:
    #     last_state_name = df_population_difference_sorted.states.iloc[-1]
    #     last_state_population = format_number(df_population_difference_sorted.population.iloc[-1])
    #     last_state_delta = format_number(df_population_difference_sorted.population_difference.iloc[-1])
    # else:
    #     last_state_name = '-'
    #     last_state_population = '-'
    #     last_state_delta = ''
    # st.metric(label=last_state_name, value=last_state_population, delta=last_state_delta)

    # st.markdown('#### States Migration')

    # donut_chart_greater = make_donut(32, 'Forecasted Churn Rate with Changes', 'green')
    # donut_chart_less = make_donut(23, 'Present Churn', 'red')

    # migrations_col = st.columns((0.2, 1, 0.2))
    # with migrations_col[1]:
    #     st.write('Inbound')
    #     st.altair_chart(donut_chart_greater)
    #     st.write('Outbound')
    #     st.altair_chart(donut_chart_less)

# Set a seed for the random number generator
np.random.seed(0)


# Hardcode values for churn_upper and churn_lower
churn_upper = 42
churn_lower = 23

# Create a list of values within the range of churn_lower and churn_upper with some variation
values = list(np.random.uniform(churn_lower, churn_upper, 52))

# Create the DataFrame
state_df = pd.DataFrame({'Values': values})
df_selected_year['population'] = state_df.values

with col[1]:
    st.markdown('#### Statistics of Churn Rate')

# Create a new figure
    if (figure_value == 0):
        fig, ax = plt.subplots()

        # Set the title and labels
        ax.set_title('Ready to generate Statistics...')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')

        # Display the figure in Streamlit
        st.pyplot(fig)
    else:
        st.image('spider_plot.png')

    # choropleth = make_choropleth(df_selected_year, 'states_code', 'population', 'magma')
    # st.plotly_chart(choropleth, use_container_width=True)

    # heatmap = make_heatmap(df_reshaped, 'year', 'states', 'population', 'magma')
    # st.altair_chart(heatmap, use_container_width=True)

    with st.expander('Insights From Andrew', expanded=True):
        chat_value = ''
        with open('chat_value.txt', 'r') as f:
            chat_value = f.read()
            print(chat_value)
            st.write(f'''
                - :orange[**Potential reasons customer might churn**]: {chat_value}
                ''')

with col[2]:
    # st.markdown('#### Top States')

    # st.dataframe(df_selected_year_sorted,
    #              column_order=("states", "population"),
    #              hide_index=True,
    #              width=None,
    #              column_config={
    #                 "states": st.column_config.TextColumn(
    #                     "States",
    #                 ),
    #                 "population": st.column_config.ProgressColumn(
    #                     "Population",
    #                     format="%f",
    #                     min_value=0,
    #                     max_value=max(df_selected_year_sorted.population),
    #                  )}
    #              )

    # st.markdown('#### Gains/Losses')

    # df_population_difference_sorted = calculate_population_difference(df_reshaped, selected_year)

    # if selected_year > 2010:
    #     first_state_name = df_population_difference_sorted.states.iloc[0]
    #     first_state_population = format_number(df_population_difference_sorted.population.iloc[0])
    #     first_state_delta = format_number(df_population_difference_sorted.population_difference.iloc[0])
    # else:
    #     first_state_name = '-'
    #     first_state_population = '-'
    #     first_state_delta = ''
    # st.metric(label='india', value=first_state_population, delta=first_state_delta)

    # if selected_year > 2010:
    #     last_state_name = df_population_difference_sorted.states.iloc[-1]
    #     last_state_population = format_number(df_population_difference_sorted.population.iloc[-1])
    #     last_state_delta = format_number(df_population_difference_sorted.population_difference.iloc[-1])
    # else:
    #     last_state_name = '-'
    #     last_state_population = '-'
    #     last_state_delta = ''
    # st.metric(label=last_state_name, value=last_state_population, delta=last_state_delta)

    st.markdown('#### Churn Forecast')

    
    donut_chart_greater = make_donut(
        churn_lower, 'Forecasted Churn Rate', 'green')
    donut_chart_less = make_donut(churn_upper, 'Present Churn Rate', 'red')

    migrations_col = st.columns((0.2, 1, 0.2))
    with migrations_col[1]:
        st.write('Forecasted Churn Rate')
        st.altair_chart(donut_chart_greater)
        st.write('Present Churn Rate')
        st.altair_chart(donut_chart_less)

    with st.expander('About', expanded=True):
        st.write('''
            - Data: [Bank data](https://www.kaggle.com/datasets/pentakrishnakishore/bank-customer-churn-data).
            - :orange[**Present Churn Rate**]: percentage of customers who have churned in the past year
            - :orange[**Future Churn Rates**]: future churn rates are forecasted based on the present churn rate
            ''')
