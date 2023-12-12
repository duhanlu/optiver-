import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from metaflow import Flow
from metaflow import get_metadata, metadata

FLOW_NAME = 'OptiverFlow'
metadata('..')
print(get_metadata())
def get_latest_successful_run(flow_name: str):
    "Gets the latest successfull run."
    for r in Flow(flow_name).runs():
        if r.successful: 
            return r
latest_run = get_latest_successful_run(FLOW_NAME)
latest_model = latest_run.data.model_0
df = pd.read_csv('validation_data0.csv')

st.title("Stock Price Movement Prediction")

st.subheader('The Ranking of Stocks',divider = 'rainbow')

# Load data from a CSV file
csv_path = 'new_test_data.csv'
data = pd.read_csv(csv_path, dtype={'stock_id': int})


# Filter data for 'seconds_in_bucket' equal to 540
filtered_data = data[data['seconds_in_bucket'] == 540]

# Date selector
selected_date = st.selectbox('Choose Date', filtered_data['date_id'].unique())

# Get data for the selected date
selected_data = filtered_data[filtered_data['date_id'] == selected_date]

# Get the Top 5 and Bottom 5 stocks
top_stocks = selected_data.nlargest(5, 'predicted_target')[['stock_id', 'predicted_target']]
bottom_stocks = selected_data.nsmallest(5, 'predicted_target')[['stock_id', 'predicted_target']]

# Convert 'stock_id' to integer
top_stocks['stock_id'] = top_stocks['stock_id'].astype(int)
bottom_stocks['stock_id'] = bottom_stocks['stock_id'].astype(int)

# Create a two-column layout
col1, col2 = st.columns(2)

def generate_table_html(df, header_color):
    # Generate an HTML table
    table_html = f"<table style='width:100%; border-collapse: collapse;'><thead style='background-color:{header_color};color:white'>"
    for col in df.columns:
        table_html += f"<th style='border: 1px solid black; padding: 8px;'>{col}</th>"
    table_html += "</thead><tbody>"
    for i, row in df.iterrows():
        table_html += "<tr>"
        for val in row:
            table_html += f"<td style='border: 1px solid black; padding: 8px; background-color: white; color: black;'>{val}</td>"
        table_html += "</tr>"
    table_html += "</tbody></table>"
    return table_html

# Display the Top 5 stocks
with col1:
    st.subheader('Top 5 Stocks')
    st.markdown(generate_table_html(top_stocks, "green"), unsafe_allow_html=True)

# Display the Bottom 5 stocks
with col2:
    st.subheader('Bottom 5 Stocks')
    st.markdown(generate_table_html(bottom_stocks, "red"), unsafe_allow_html=True)

st.subheader('Input your single data entry to get prediction!', divider='rainbow')
stockid = st.number_input('Insert stock for predicion (Must be an integer)', value = 1.00)
dateid = st.number_input('Insert date for predicion (Must be an integer)',value = 478.00)
second_in_bucket = st.number_input('Insert second for predicion (Must be an integer)',value = 0.00)
imbalance_size = st.number_input('Insert imbalance size for predicion',value = 1019888.25)
imbalance_buy_sell_flag = st.number_input('Insert imbalance buy sell flag for predicion',value = 1)
reference_price = st.number_input('Insert reference price for predicion',value = 0.999988)
matched_size = st.number_input('Insert matched size for predicion',value = 2913478.86)
far_price = st.number_input('Insert far price for predicion',value = 0)
near_price = st.number_input('Insert near price for predicion',value = 0)
bid_price = st.number_input('Insert bid price for predicion',value = 0.999795)
bid_size = st.number_input('Insert bid size for predicion',value = 22231)
ask_price = st.number_input('Insert ask price for predicion',value = 1.000182)
ask_size = st.number_input('Insert ask size for predicion',value = 19705.32)
wap = st.number_input('Insert wap for predicion',value = 1)
features = [
                'seconds_in_bucket', 'imbalance_buy_sell_flag',
                'imbalance_size', 'matched_size', 'bid_size', 'ask_size',
                'reference_price', 'far_price', 'near_price', 'ask_price', 'bid_price', 'wap',
                'imb_s1', 'imb_s2'
        ]
def calculate_additional_price_features(df, features):
        # Define a list of price-related columns
        prices = ['reference_price', 'far_price', 'near_price', 'ask_price', 'bid_price', 'wap']

        # Loop through the price columns to create new features
        for i, a in enumerate(prices):
            for j, b in enumerate(prices):
                for k, c in enumerate(prices):
                    # Check if the order of prices a, b, and c is descending
                    if i > j and j > k:
                        # Calculate the maximum, minimum, and mid values among a, b, and c
                        max_ = df[[a, b, c]].max(axis=1)
                        min_ = df[[a, b, c]].min(axis=1)
                        mid_ = df[[a, b, c]].sum(axis=1) - min_ - max_

                        # Calculate and add a new feature to the DataFrame
                        df[f'{a}_{b}_{c}_imb2'] = (max_ - mid_) / (mid_ - min_)
                        # Add the new feature name to the list of features
                        features.append(f'{a}_{b}_{c}_imb2')

        # Return the modified DataFrame and the updated list of features
        return df, features
def calculate_imbalance_features(df):
        df['imb_s1'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'])
        df['imb_s2'] = (df['imbalance_size'] - df['matched_size']) / (df['matched_size'] + df['imbalance_size'])
        return df
def calculate_price_features(df, features):
        # Define a list of price-related columns
        prices = ['reference_price', 'far_price', 'near_price', 'ask_price', 'bid_price', 'wap']

        # Loop through the price columns to create new features
        for i, a in enumerate(prices):
            for j, b in enumerate(prices):
                # Check if the first price (a) comes after the second price (b) in the list
                if i > j:
                    # Calculate and add a new feature to the DataFrame
                    df[f'{a}_{b}_imb'] = df.eval(f'({a} - {b}) / ({a} + {b})')
                    # Add the new feature name to the list of features
                    features.append(f'{a}_{b}_imb')

        # Return the modified DataFrame and the updated list of features
        return df, features

def generate_features(df, features):
            # Calculate imbalance features
            df = calculate_imbalance_features(df)  # 📊 Calculate imbalance features
            # Calculate features based on price differences
            df, features = calculate_price_features(df, features)  # 💰 Calculate price-related features
            # Calculate additional features based on price differences
            df, features = calculate_additional_price_features(df, features)  # 🔄 Calculate additional price features
            # Return the DataFrame with selected features

            return df,features

def add_new_feature(df, features):
        #1 absolute privce
        df['abs_far_p_dis'] = abs(df.far_price - df.reference_price)
        df['abs_far_p_dis'].fillna(0, inplace = True)
        df['abs_bid_p_dis'] = abs(df.bid_price - df.reference_price)
        df['abs_bid_p_dis'].fillna(0, inplace = True)

        # relative price difference
        df['rel_far_p_dis'] = (df.far_price - df.reference_price)/df.reference_price
        df['rel_far_p_dis'].fillna(0, inplace = True)

        df['rel_bid_p_dis'] = (df.bid_price - df.reference_price)/df.reference_price
        df['rel_bid_p_dis'].fillna(0, inplace = True)
        list_abs_value = ['abs_far_p_dis','abs_bid_p_dis','rel_far_p_dis','rel_bid_p_dis']
        features.extend(list_abs_value)

        #2.2 volume change
        df['volume'] = df.bid_size + df.ask_size
        df['Volume_change_rate'] = df.groupby(['date_id','stock_id'])['volume'].pct_change()
        features.append('Volume_change_rate')

        #2.3 reference moving avg
        window_size = 6  
        df['reference_moving_avg'] = df['reference_price'].rolling(window=window_size).mean()
        features.append('reference_moving_avg')

        #3 bid-adk pressure
        def calculate_pressure_ratio(group):
            weighted_ask_sum = (group['ask_size'] * group['ask_price']).sum()
            weighted_bid_sum = (group['bid_size'] * group['bid_price']).sum()
            return weighted_ask_sum / weighted_bid_sum if weighted_bid_sum != 0 else None
        # Group by stock and apply the calculation
        pressure_by_stock = df.groupby(['stock_id','date_id']).apply(calculate_pressure_ratio)
        # Match the index
        pressure_by_stock = pressure_by_stock.reindex(df.set_index(['stock_id', 'date_id']).index, fill_value=None)
        # Reset the index if necessary
        df['pressure_by_stock'] = pressure_by_stock.values
        features.append('pressure_by_stock')

        #4 wap change
        df['wap_change']=  df.groupby(['stock_id','date_id'])['wap'].pct_change()
        features.append('wap_change')
        #df_new = df_.groupby(['date_id', 'stock_id'])['reference_price'].shift(0) - df_.groupby(['date_id', 'stock_id']).shift(1)['reference_price']
        #df_['price_flag'] = df_new.dropna() >= 0


        #5 RSI
        def calculate_rsi(data, window=6):
            delta = df.groupby(['stock_id', 'date_id'])['wap'].diff(1).dropna()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)

            # Calculate the average gain and loss
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            # Calculate the relative strength (RS)
            rs = avg_gain / avg_loss
            # Calculate the RSI
            rsi = 100 - (100 / (1 + rs))
            return rsi

        # Calculate RSI on the 'wap' column
        df['RSI'] = calculate_rsi(df['wap'])
        features.append('RSI')


        #6 B_index
        def calculate_B_index(df):
            B = (df.ask_size - df.bid_size)/(df.ask_size +df.bid_size)
            B_star = B * np.log(1 + df.bid_size + df.ask_size)
            return B_star
        df['B_star'] = calculate_B_index(df)
        features.append('B_star')

        return df, features
single_row = {"stock_id": stockid, "date_id": dateid,"seconds_in_bucket": second_in_bucket, "imbalance_size": imbalance_size,"imbalance_buy_sell_flag":imbalance_buy_sell_flag,
              "reference_price": reference_price,"matched_size": matched_size,"far_price": far_price,"near_price":near_price,
              "bid_price":bid_price,"bid_size": bid_size,"ask_price":ask_price,"ask_size":ask_size,"wap":wap}
index_value = [0]
single_row_df = pd.DataFrame(single_row, index=[index_value])

single_row_data, features = generate_features(single_row_df,features)
single_row_data,features = add_new_feature(single_row_data,features)
single_row_np = single_row_data.values
prediction = latest_model.predict(single_row_np)
st.write("The prediction is: ", prediction)

st.subheader('Upload your dataset to get a bunch of predictions at same time!', divider='rainbow')
uploaded_file = st.file_uploader("Choose a csv file")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

    uploaded_df, features_ = generate_features(dataframe,features)
    uploaded_df, features_ = add_new_feature(uploaded_df,features_)
    uploaded_np = uploaded_df.values
    predictions = latest_model.predict(uploaded_np)
    st.write("The predictions are: ", predictions)


st.subheader('Select a stock to check how does it change recently!', divider='rainbow')
input = st.number_input("Insert Stock ID", value=1, placeholder="Type stock ID...")
st.write('The current selection: ', input)

value_to_filter = [0,100,200,300,400,500]
selected_rows = df[df['stock_id'] == input]
selected_rows = selected_rows[selected_rows['date_id'] == 479]
#selected_rows = selected_rows[selected_rows['seconds_in_bucket'].isin(value_to_filter)]
print(selected_rows)
chart_data = pd.DataFrame(selected_rows,columns=['seconds_in_bucket','predictions','revealed_target'])

st.line_chart(chart_data,x='seconds_in_bucket',y=['predictions'])