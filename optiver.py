import pandas as pd       
import numpy as np   
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error                  
import os    

#### load dataset #####
df = pd.read_csv('train.csv')
df_train = df.iloc[:10506]
test_x_df = pd.read_csv('example_test_files/test.csv')
test_y_df = pd.read_csv('example_test_files/revealed_targets.csv')
test_x_df = test_x_df.iloc[:10506]
test_y_df = test_y_df.iloc[:10506]

#### preprocess data ####
def calculate_imbalance_features(df):
    # 📈 Calculate and add imbalance feature 1 (imb_s1)
    df['imb_s1'] = df.eval('(bid_size - ask_size) / (bid_size + ask_size)')  

    # 🔃 Calculate and add imbalance feature 2 (imb_s2)
    df['imb_s2'] = df.eval('(imbalance_size - matched_size) / (matched_size + imbalance_size)') 

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

def generate_features(df):
    # Define the list of feature column names
    features = ['seconds_in_bucket', 'imbalance_buy_sell_flag',
                'imbalance_size', 'matched_size', 'bid_size', 'ask_size',
                'reference_price', 'far_price', 'near_price', 'ask_price', 'bid_price', 'wap',
                'imb_s1', 'imb_s2'
               ]
    
    # Calculate imbalance features
    df = calculate_imbalance_features(df)  # 📊 Calculate imbalance features
    
    # Calculate features based on price differences
    df, features = calculate_price_features(df, features)  # 💰 Calculate price-related features
    
    # Calculate additional features based on price differences
    df, features = calculate_additional_price_features(df, features)  # 🔄 Calculate additional price features
    
    # Return the DataFrame with selected features
    return df,df[features],features
features = ['seconds_in_bucket', 'imbalance_buy_sell_flag',
                'imbalance_size', 'matched_size', 'bid_size', 'ask_size',
                'reference_price', 'far_price', 'near_price', 'ask_price', 'bid_price', 'wap',
                'imb_s1', 'imb_s2'
               ]
#train_df_check_3 = calculate_additional_price_features(train_df_check_2)

## process training data
processed__train_df, X_train_df,features = generate_features(df_train)
processed__train_df.fillna(0, inplace=True)
processed__train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
processed__train_df.dropna(axis=0, inplace=True)


X_train = processed__train_df[features].values
Y_train = processed__train_df['target'].values

### process test data
processed_test_df, X_test_df,features = generate_features(test_x_df)
X_test_df.fillna(0, inplace=True)
Y_test_df = test_y_df['revealed_target']
test_df = X_test_df.join(Y_test_df)

test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.dropna(axis=0, inplace=True)

X_test = test_df[features].values
Y_test = test_df['revealed_target'].values

# Create an index array for data splitting
index = np.arange(len(X_train))

### train model 
model = LinearRegression()
model.fit(X_train, Y_train)


### predict model
y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, y_pred)
print(f'Mean Squared Error: {mse}')