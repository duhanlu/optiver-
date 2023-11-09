# optiver-
Train a model that can predict stock market price movement

Phase I:
Duplicate a model that is similar with the notebook. Train the model with LGBMRegressor, XGBRegressor, and CatBoostRegressor firstly. And compared which model is best
based on features calculated in the notebook.
• Done with the data preprocess. Chosen 10560 data based on the observation of the far price and near price with/without values apearing alternately. Then the subdateset contains 
data in a period(half has far price and near price, another hald doesn't).
• Built a simple sklearn regression learning with MSE: 18633.986463027177. This model definately is a bad one. So then we should try out other advanced model.

- problems:
• The column with nan value, far price and near price, appears periodly. ----- (1). Need to find out the meaning of it and how to deal with it?
• it contains inf number after calculating the other features.  ------ is that ok to just remove them?
- Suggestions on tracking model preformance:
• Use Comet to track every training process and record the model performance, for being easier to pick up a better model later.

Phase II:
Research on financial model to create new training features to improve model accuracy. 

