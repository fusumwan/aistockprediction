
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats
from copy import copy
import pandas as pd
import plotly.figure_factory as ff
import plotly.io as pio
from tensorflow import keras
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
import SimulatedAnnealing as SA_AI
import funcInterFace as FI
import ML_Stock as ML
import RidgeAI as RI
import LSTMAI as LS
import random
'''
///-- =============================================
///-- Author: Sum Wan,FU
///-- Create date: 2021-10-21
///-- Student ID: a1714470
///-- Course:Applied Machine Learning (merged 4416_4816_7416)
///-- The main function is "MLPrediction.py", user need to type the following command to run this program. 
///-- python3 MLPrediction.py;
///-- This command will load both "stock.csv" and "stock_volume.csv" files and then it will generate different chat graphics for different stock. At least it will export the "Result1.txt" which include all accuracy results.
///---There many graphics are exported into several folders, such as "Average","Ridge","LSTM", and "Original". For instance, the stock price charts of the 9 companies from January 12, 2012 to August 11, 2020 are all placed in the "original" folder. The prediction results of the Ridge regression model and LSTM model will be saved in "Ridge" and "LSTM" respectively. Finally, the "average" folder contains all the prediction results using the effective averaging method.
///-- Need to install some package: pip install scipy, pip3 install scipy, pip install plotly,pip3 install plotly,pip install numpy,pip3 install numpy,python -m pip install -U pip ,python -m pip install -U matplotlib,python3 -m pip3 install -U pip,python3 -m pip3 install -U matplotlib,pip install pandas,pip3 install pandas,pip install --upgrade tensorflow,pip3 install --upgrade tensorflow,pip install scikit-learn,pip3 install scikit-learn,pip install -U kaleido,pip3 install -U kaleido
///-- =============================================
'''





if __name__ == '__main__':
	dx=0.0001
	StockDetails = [['AAPL', 0.0 ,[],[],[],[],0.0,0.0,0.0], ['BA', 0.0,[],[],[],[],0.0,0.0,0.0], ['T', 0.0 ,[],[],[],[],0.0,0.0,0.0], ['MGM', 0.0 ,[],[],[],[],0.0,0.0,0.0],['AMZN', 0.0 ,[],[],[],[],0.0,0.0,0.0], ['IBM', 0.0 ,[],[],[],[],0.0,0.0,0.0], ['TSLA', 0.0 ,[],[],[],[],0.0,0.0,0.0],['GOOG', 0.0 ,[],[],[],[],0.0,0.0,0.0],['sp500', 0.0 ,[],[],[],[],0.0,0.0,0.0]]
	ML=ML.ML_Stock(dx,False,"stock.csv","stock_volume.csv")
	RidgeControl=RI.RidgeAI(dx,False,"stock.csv","stock_volume.csv")
	LSTMControl=LS.LSTMAI(dx,False,"stock.csv","stock_volume.csv")

	SA=SA_AI.SimulatedAnnealing(dx)
	ML.print_original_result()
	p_stockPrice=ML.get_stockPriceData()
	for i in range(len(StockDetails)):
		lstm_df_predicted=[]
		df_predicted=[]
		stock_name=StockDetails[i][0]

		lstm_predicted=LSTMControl.lstm_training(stock_name)
		ridge_predicted=RidgeControl.ridge_regression_training(stock_name)
		ML.set_ridgePredicted(ridge_predicted)
		ML.set_lstmPredicted(lstm_predicted)
		ridge_acc=ML.rss(np.array(ridge_predicted['predictions']),np.array(lstm_predicted['Close']))
		lstm_acc=ML.rss(np.array(lstm_predicted['predictions']),np.array(lstm_predicted['Close']))
		SAFunc=FI.funcInterFace(np.array(lstm_predicted['Close']),np.array(ridge_predicted['predictions']),np.array(lstm_predicted['predictions']),ridge_acc,lstm_acc)
		StockDetails[i][1]=SA.hillClimbingCalMax(SAFunc)
		StockDetails[i][2]=ML.get_averagePredicted(SA.getPercentage())
		StockDetails[i][3]=RidgeControl.get_ridgePredicted()
		StockDetails[i][4]=LSTMControl.get_lstmPredicted()
		StockDetails[i][5]=ridge_acc
		StockDetails[i][6]=lstm_acc
		StockDetails[i][7]=SA.getAccuracy()



		RidgeControl.print_ridge_regression_result(stock_name)
		LSTMControl.print_lstm_result(stock_name)
		ML.print_result(stock_name)
		ML.print_stock_original_result(stock_name)


	f = open("Result1.txt", "w")
	
	for i in range(len(StockDetails)):
		stock_name=StockDetails[i][0]
		ridge=StockDetails[i][5]
		lstm=StockDetails[i][6]
		avg=StockDetails[i][7]
		message=(stock_name+"  Effective Avg accuracy:"+str(avg) + "  Ridge accuracy:" + str(ridge)+ "  LSTM accuracy:" + str(lstm))+"\n"
		print(message)
		f.write(message)
	f.close()


