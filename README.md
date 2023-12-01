# Prediction of stock closing price movement
## Objective
The objective of this project is to make prediction of the stock closing price movement for companies to make investment decisions.
## Description
The dataset is from Kaggle, containing 1 million entries. It contains market indexes such as bid price, ask, price, bid size, ask size, reference price, matched size etc. The target is the final price movement direction with representation of the sign and the movement magnitude with representation of the absolute value of the target. The target is calculated from financial model with WAP index. So for the users, they can input the current market index, to get the predicted closing price trend respective to the current time. 
## Instruction
We've fit the model into metaflow in order to get all data saved. The comet is used to get track of the performance and get visualization of the performance. The user interface is designed by streamlit. Therefore, there is a file called optiver.py, for training and testing models with metaflow as the structure. User can use the user interface we designed to get prediction, which is programmed in the file called prediction.py.
## How to get 
* Directly run prediction.py by using command: streamlit prediction.py run to open the user interface.
* input index for demonstration to get the prediction by clicking submit bottom.
  
