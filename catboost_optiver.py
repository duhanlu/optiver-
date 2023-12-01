import pandas as pd       
import numpy as np   
from comet_ml import Experiment 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
import catboost as cbt  
from sklearn.metrics import r2_score  
import math      
import os    
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

#### load dataset #####
df_train = pd.read_csv('train.csv')
test_x_df = pd.read_csv('example_test_files/test.csv')
test_y_df = pd.read_csv('example_test_files/revealed_targets.csv')
## if we want to test a small dataset
df_train = df_train.iloc[:10506]
test_x_df = test_x_df.iloc[:10506]
test_y_df = test_y_df.iloc[:10506]



#### add features #####
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
    return df,features

def add_new_feature(df,features):
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
    #2.2 volume chanve
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

    return df,features

selected_features = ['seconds_in_bucket', 'imbalance_buy_sell_flag',
                'imbalance_size', 'matched_size', 'bid_size', 'ask_size',
                'reference_price', 'far_price', 'near_price', 'ask_price', 'bid_price', 'wap',
                'imb_s2','imbalance_buy_sell_flag','near_price_reference_price_imb','far_price_reference_price_imb',
                'bid_price_ask_price_reference_price_imb2','ask_price_near_price_far_price_imb2','near_price_far_price_reference_price_imb2',
                'ask_price_near_price_reference_price_imb2','ask_price_far_price_reference_price_imb2','wap_bid_price_ask_price_imb2',
                'wap_ask_price_near_price_imb2','wap_bid_price_near_price_imb2','abs_bid_p_dis','rel_far_p_dis','Volume_change_rate']
## process training data
processed__train_df, features= generate_features(df_train)
processed__train_df.fillna(0, inplace=True)
processed__train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
processed__train_df.dropna(axis=0, inplace=True)

second_processed_train_df, features = add_new_feature(processed__train_df,features)
second_processed_train_df.to_csv("data_for_corr_unnormal.csv",index=False)
### process test data
X_train = second_processed_train_df[selected_features].values
Y_train = second_processed_train_df['target'].values
print("current features: ", features)

processed_test_df,features= generate_features(test_x_df)
processed_test_df.fillna(0, inplace=True)
Y_test_df = test_y_df['revealed_target']
processed_test_df = processed_test_df.join(Y_test_df)

processed_test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
processed_test_df.dropna(axis=0, inplace=True)

second_processed_test_df, features = add_new_feature(processed_test_df,features)

X_test = second_processed_test_df[selected_features].values
y_test = second_processed_test_df['revealed_target'].values
print("Dimension of X test: ", X_test.shape)
print("Dimension of Y test: ", y_test.shape)
test_data = np.column_stack((X_test, y_test))


### train model 



model = cbt.CatBoostRegressor()
grid = {'learning_rate': [0.03, 0.1,10^(-6)],
        'depth': [1, 2, 3],
        'l2_leaf_reg': [1, 2, 3],
        'iterations': [20,30]}
"""grid_search_result = model.grid_search(grid,
                                       X=X_train,
                                       y=Y_train,
                                       plot=True)"""

grid_search = GridSearchCV(model, grid, cv=2, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)
best_model = grid_search.best_estimator_
test_predictions = best_model.predict(X_test)
test_mae = np.mean(np.abs(test_predictions - y_test))
print("Test mae:", test_mae)


model = cbt.CatBoostRegressor(
        objective='MAE',
        iterations=best_params['iteration'],         # Higher may capture more complex patterns but may overfit
        learning_rate=best_params['learning_rate'],      # Lower for stability, but may require more iterations
        depth=best_params['depth'],                 # Higher may capture more complex patterns but may overfit
        l2_leaf_reg=best_params['l2_leaf_reg'],           # Adjust as needed, higher for more regularization
        verbose=10
    )
model.fit(X_train, Y_train, 
                  eval_set=[(X_test, y_test)], 
                  verbose=10, 
                  early_stopping_rounds=100
                 )


best_iteration = model.get_best_iteration()
best_mae = model.get_best_score()['validation']['MAE']
metric = {'best iteration': best_iteration, 'best mae': best_mae}
print('Test Performance')
print('mae: {:.2f}'.format(best_mae))

