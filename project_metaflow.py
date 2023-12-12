#!/usr/bin/env python

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
from io import StringIO 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from metaflow import FlowSpec, step, IncludeFile, current


"""
Here we have three files: 
1. train.csv is used to train the model
2. test.csv is the used to do the prediciton
3. reveal_targets.csv contains the true values which help us to evaluate the performance(mae)
"""

class OptiverFlow(FlowSpec):

    train_file = IncludeFile(
        'df_train',
        help = 'CSV file with the training dataset',
        default = 'train.csv'
    )

    test_x_file = IncludeFile(
        'test_x_df',
        help = 'CSV file with the test dataset',
        default = 'test.csv'
    )

    test_y_file = IncludeFile(
        'test_y_df',
        help = 'CSV file with the revealed target dataset',
        default = 'revealed_targets.csv'
    )
    assertion_data = IncludeFile(
        'assertion_data_df',
        help = 'CSV file with the certain data rows for qualitative test',
        default = 'assertion_data.csv'
    )



    @step
    def start(self):
        """
        Start up and print out some info to make sure everything is ok metaflow-side
        """
        print("Starting up at {}".format(datetime.utcnow()))
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        self.next(self.load_data)



    @step
    def load_data(self):
        """
        Read the training and serving data in from the included files.
        """
        # Convert the included file content from bytes to a file-like object
        self.df_train = pd.read_csv(StringIO(self.train_file))
        self.test_x_df = pd.read_csv(StringIO(self.test_x_file))
        self.test_y_df = pd.read_csv(StringIO(self.test_y_file))
        self.assertion_data_df = pd.read_csv(StringIO(self.assertion_data))
        corr_matrix = self.df_train.corr()
        heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        heatmap.get_figure().savefig('heatmap.png')
        #plt.show()
        '''## if we want to test a small dataset
        self.df_train = self.df_train.iloc[:10506]
        self.df_x_test = self.df_x_test.iloc[:10506]
        self.df_y_test = self.df_y_test.iloc[:10506]
        '''
        # Go to the next step
        self.next(self.split_data)


    @step
    def split_data(self):
        """
        Step 1:
        We need to check that the tainning set and the testing set has not overlapping parts
        """
        self.df_train = self.df_train.drop(self.df_train[self.df_train['date_id'] >= 478].index)

        """
        Step 2:
        Here comes one of the most important part of the dataset processing
        The acution is splited into two parts: 
        3:50pm -3:55pm (5min, 300seconds in total);
        3:55pm - 4:00pm (5min, 300 seconds in total)
        In the first 5 minutes, no far price or far price are announced.

        Thus, to better do the prediction, we need to plit the dataset into two(_0 and _1),
        otherwise, there would be many Nan columns when creating features.
        """
        def add_time_flag(df, df_name):
            df['time_flag'] = 0
            df.loc[df.seconds_in_bucket > 290, 'time_flag'] = 1  
            df.loc[df.seconds_in_bucket <= 290, 'time_flag'] = 0  
            df1 = df.loc[df.seconds_in_bucket > 290]
            df0 = df.loc[df.seconds_in_bucket <= 290]
            
            return {f"{df_name}_0": df0, f"{df_name}_1": df1}

        # train dataset
        self.result_dfs_train = add_time_flag(self.df_train, "df_train")
        # rename and split the dataset into two dataframes
        self.df_train_0 = self.result_dfs_train["df_train_0"]
        self.df_train_1 = self.result_dfs_train["df_train_1"]

        # test dataset
        self.reseult_dfs_test = add_time_flag(self.test_x_df, "test_x_df")
        # rename and split the dataset into two dataframes
        self.test_x_df_0 = self.reseult_dfs_test["test_x_df_0"]
        self.test_x_df_1 = self.reseult_dfs_test["test_x_df_1"]

        self.next(self.feature_calculation_functions)


    
    @step
    def feature_calculation_functions(self):
        """
        In these following first 3 functions, we especially creates imbalanced functions 
        which are obtained by paring the different columns.
        """
        
        self.next(self.define_feature_list)


    def calculate_imbalance_features(self,df):
        df['imb_s1'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'])
        df['imb_s2'] = (df['imbalance_size'] - df['matched_size']) / (df['matched_size'] + df['imbalance_size'])
        return df

    def calculate_price_features(self, df, features):
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
    
    def calculate_additional_price_features(self, df, features):
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

    def generate_features(self, df, features):
            # Calculate imbalance features
            df = self.calculate_imbalance_features(df)  # 📊 Calculate imbalance features
            # Calculate features based on price differences
            df, features = self.calculate_price_features(df, features)  # 💰 Calculate price-related features
            # Calculate additional features based on price differences
            df, features = self.calculate_additional_price_features(df, features)  # 🔄 Calculate additional price features
            # Return the DataFrame with selected features

            return df,features

    """
    Here we add other features in wish to elevate the model performance.
    The features include: 
    absolute price, volume change ratio, reference moving average, bid-ask pressure ratio,
    wap change ratio, RSI(relative strength ratio), B_index(another way to evatuate the imbalance of bid&ask)
    """
    def add_new_feature(self, df, features):
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


    @step
    def define_feature_list(self):
        # Define the features
        self.features = [
                'seconds_in_bucket', 'imbalance_buy_sell_flag',
                'imbalance_size', 'matched_size', 'bid_size', 'ask_size',
                'reference_price', 'far_price', 'near_price', 'ask_price', 'bid_price', 'wap',
                'imb_s1', 'imb_s2'
        ]

        # Define the selected features for the dataset _1
        self.selected_features = [
            'seconds_in_bucket', 'imbalance_buy_sell_flag', 'imbalance_size', 'matched_size',
            'bid_size', 'ask_size', 'reference_price', 'far_price', 'near_price', 'ask_price',
            'bid_price', 'wap', 'imb_s2', 'near_price_reference_price_imb',
            'far_price_reference_price_imb', 'bid_price_ask_price_reference_price_imb2',
            'ask_price_near_price_far_price_imb2', 'near_price_far_price_reference_price_imb2',
            'ask_price_near_price_reference_price_imb2', 'ask_price_far_price_reference_price_imb2',
            'wap_bid_price_ask_price_imb2', 'wap_ask_price_near_price_imb2',
            'wap_bid_price_near_price_imb2' , 'Volume_change_rate', 'pressure_by_stock','B_star'
        ]

        # Define the delected features for the dataset _0
        # Here we deleted all the features derived from near_price or far_price
        self.selected_features_0 = [
            'seconds_in_bucket', 'imbalance_buy_sell_flag',
            'imbalance_size', 'matched_size', 'bid_size', 'ask_size',
            'reference_price', 'ask_price', 'bid_price', 'wap',
            'imb_s2','imbalance_buy_sell_flag','bid_price_ask_price_reference_price_imb2',
            'wap_bid_price_ask_price_imb2','abs_bid_p_dis','Volume_change_rate', 'pressure_by_stock','B_star'
                ]

        print("Your features are all set!")

        self.next(self.process_data)


    @step
    def process_data(self):
        def train_df_processing(train_df,features,selected_features):
            
            # original features
            processed__train_df, features= self.generate_features(train_df, features)
            
            # save the "near_price" and "far_price" columns
            near_far_columns = processed__train_df[['near_price', 'far_price']].copy()
            # remove these two columns from the original dataframe
            processed__train_df = processed__train_df.drop(['near_price', 'far_price'], axis=1)

            # fillna, replace and dropna for the rest of the columns
            processed__train_df.fillna(0, inplace=True)
            processed__train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            #processed__train_df.dropna(axis=0, inplace=True)
            
            # join the processed dataframe with these two columns
            processed__train_df = processed__train_df.join(near_far_columns)

            # add new features
            second_processed_train_df, features = self.add_new_feature(processed__train_df,features)
            #print(second_processed_train_df.columns)
            second_processed_train_df.to_csv("data_for_corr_unnormal.csv",index=False)

            X_train = second_processed_train_df[selected_features].values
            Y_train = second_processed_train_df['target'].values
            print("current features: ", features)

            return X_train, Y_train
            #   used for gridsearch hyperparameter tuning & train model


        def test_df_processing(test_x_df,features,selected_features): 
            # generate features
            processed_test_df, features = self.generate_features(test_x_df, features)
            Y_test_df = self.test_y_df['revealed_target']
            processed_test_df = processed_test_df.join(Y_test_df)
            print(processed_test_df)
            
            # save the "near_price" and "far_price" columns
            near_far_columns = processed_test_df[['near_price', 'far_price']].copy()
            # remove these two columns from the original dataframe
            processed_test_df = processed_test_df.drop(['near_price', 'far_price'], axis=1)

            # fillna, replace and sropna for the rest of the columns
            processed_test_df.fillna(0, inplace=True)
            processed_test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            #processed_test_df.dropna(axis=0, inplace=True)
            """do not use this because: the near_price_far_price_ibm2 is nan but not in the selected_feature list.
                Then it would not be in the X_test list"""
            # rejoin the near and far price columns back to the dataframe
            processed_test_df = processed_test_df.join(near_far_columns)

            second_processed_test_df, features = self.add_new_feature(processed_test_df, features)
            #print(second_processed_test_df.columns)

            X_test = second_processed_test_df[selected_features].values
            y_test = second_processed_test_df['revealed_target'].values
            print("Dimension of X test: ", X_test.shape)
            print("Dimension of Y test: ", y_test.shape)
            #test_data = np.column_stack((X_test, y_test))

            return X_test, y_test, second_processed_test_df


        self.X_train, self.Y_train = train_df_processing(self.df_train_1,self.features,self.selected_features)
        self.X_test, self.y_test, self.second_processed_test_df = test_df_processing(self.test_x_df_1,self.features,self.selected_features)

        self.X_train_0, self.Y_train_0 = train_df_processing(self.df_train_0,self.features,self.selected_features_0)
        self.X_test_0, self.y_test_0 ,self.second_processed_test_df_0 = test_df_processing(self.test_x_df_0,self.features,self.selected_features_0)
        assertion_data, features = self.generate_features(self.assertion_data_df,self.features)
        assertion_data,features = self.add_new_feature(self.assertion_data_df,self.features)
        self.assertion_data_np = assertion_data.values
    
        print('You have already processed the training and testing datasets!')
        
        self.next(self.train_predict)



    @step
    def train_predict(self):
        """
        Train the model Catboost using GridSearchCV to find out the best parameters
        Then use the best parameters find to do the prediction and evalustion
        """
        def GSC(X_train, Y_train, X_test, y_test, flag=0, best_params=None):
            if flag == 1:
                model = cbt.CatBoostRegressor()
                grid = {'learning_rate': [0.03, 0.1, 10**(-6)],
                        'depth': [1, 2, 3],
                        'l2_leaf_reg': [1, 2, 3],
                        'iterations': [20, 30]}

                grid_search = GridSearchCV(model, grid, cv=2, scoring='neg_mean_absolute_error')
                grid_search.fit(X_train, Y_train)
                best_params = grid_search.best_params_
                print("Best Hyperparameters:", best_params)
                model = grid_search.best_estimator_
                
                
            else:
                if best_params is None:
                    raise ValueError("best_params must be provided if flag is not 1")
                
                model = cbt.CatBoostRegressor(
                    #task_type="GPU",
                    objective='MAE',
                    iterations=best_params['iterations'],
                    learning_rate=best_params['learning_rate'],
                    depth=best_params['depth'],
                    l2_leaf_reg=best_params['l2_leaf_reg'],
                    verbose=10
                )
            
            model.fit(X_train, Y_train, 
                    eval_set=[(X_test, y_test)], 
                    verbose=10, 
                    early_stopping_rounds=100)

            test_predictions = model.predict(X_test)                
            test_mae = np.mean(np.abs(test_predictions - y_test))
            print("Test MAE:", test_mae)
            print("Test Predictions:", test_predictions)

            return model, test_predictions, best_params, test_mae


        ## _1 dataset 
        self.best_params = {'depth': 3, 'iterations': 30, 'l2_leaf_reg': 1, 'learning_rate': 0.1}
        self.model, self.test_predictions, self.best_params, self.test_mae =  GSC(
                    self.X_train, self.Y_train, 
                    self.X_test, self.y_test, 
                    flag=0, 
                    best_params=self.best_params
                )
        
        ##_0 dataset
        self.best_params_0 = {'depth': 3, 'iterations': 30, 'l2_leaf_reg': 1, 'learning_rate': 0.1}
        self.model_0, self.test_predictions_0, self.best_params_0,self.test_mae_0 =  GSC(
            self.X_train_0, self.Y_train_0, 
            self.X_test_0, self.y_test_0, 
            flag=0, 
            best_params=self.best_params_0)
        
        def qualitative_test():
            first_row = self.assertion_data_np[0, :]
            second_row = self.assertion_data_np[1,:]
            third_row = self.assertion_data_np[2,:]
            predict_assertion = self.model_0.predict(first_row)
            print(predict_assertion)
            assert self.model_0.predict(first_row)<0
            assert self.model_0.predict(second_row)<0
            assert self.model_0.predict(third_row)<0

            return "Qualitative test've printed"
        print(qualitative_test())
        # Proceed to the next step
        self.next(self.upload_to_comet)
    

    @step
    def upload_to_comet(self):
        self.combined_test_predictions = np.concatenate([self.test_predictions, self.test_predictions_0])
        self.combined_y_test = np.concatenate([self.y_test, self.y_test_0])
        self.combined_test_mae = np.mean(np.abs(self.combined_test_predictions - self.combined_y_test))
        print('test mae: {:.4f}'.format(self.test_mae))

        def upload_to_comet(test_predictions_0, test_predictions, test_mae):
            try:
                # Create an experiment with your api key
                experiment = Experiment(
                    api_key="ctExkjTwIJfnErIdvtYxaNJ0q",
                    project_name="sz4485nyu-comet-test",
                    workspace="nyu-fre-7773-2021",
                )

                # Log the combined MAE with a descriptive context
                experiment.log_metric("mae_combined", self.combined_test_mae)
                experiment.log_metric('mae_for_first_5min', self.test_mae_0)
                experiment.log_metric('mae_for_last_5min', self.test_mae)

                # Add tags, parameters, and other relevant information
                experiment.add_tags(["Data Splitting", "MAE Comparison"])
                experiment.log_parameters({
                    "data_splitting_method": "01 scenario data splitting",
                    "model_name": "catboost",
                    "evaluation_metric": "MAE",
                })

                # End the Comet.ml experiment
                experiment.end()
                print("MAE values uploaded to comet successfully!")

            except Exception as e:
                print(f"Failed to upload to comet: {e}")
        

        # Upload the MAE values to Comet.ml
        upload_to_comet(self.test_predictions_0, self.test_predictions, self.test_mae)

        self.next(self.end)

    @step
    def end(self):
        
        combined_test_predictions = np.concatenate([self.test_predictions, self.test_predictions_0])
        combined_y_test = np.concatenate([self.y_test, self.y_test_0])
        test_mae = np.mean(np.abs(combined_test_predictions - combined_y_test))
        print('test mae: {:.4f}'.format(test_mae))
        print('Problem solved!')


if __name__ == '__main__':
    OptiverFlow()
