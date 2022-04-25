
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
from keras import backend as K
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
import MLbase as AI
import MLGraphic as MLG
import ML_Stock as ML
from sklearn.metrics import mean_squared_error
'''
///-- =============================================
///-- Author: Sum Wan,FU
///-- Create date: 2021-10-21
///-- Student ID: a1714470
///-- Course:Applied Machine Learning (merged 4416_4816_7416)
///-- This class include Ridge regression function
///-- =============================================
'''
class RidgeAI(ML.ML_Stock):

	def __init__(self, dx, isShowFigure, filename1,filename2):
		super().__init__(dx, isShowFigure, filename1,filename2)

	def ridge_regression_training(self,stock_name):
		self.get_individual_stock_prices_volume(stock_name)
		self.get_individual_stock_prices_volume_target()
		self.remove_price_volume_null_value()
		self.scale_the_date()
		self.creating_feature_and_target()
		self.spliting_data()
		
		# Note that Ridge regression performs linear least squares with L2 regularization.
		# Create and train the Ridge Linear Regression  Model
		regression_model = Ridge()
		regression_model.fit(self.XTrainData, self.YTrainData)
		# Test the model and calculate its accuracy 
		lr_accuracy = regression_model.score(self.XTestData, self.YTestData)
		#print("Ridge Regression Score: ", lr_accuracy)
		#self.ridge_acc=lr_accuracy
		# Make Prediction
		predicted_prices = regression_model.predict(self.X)
		#predicted_prices

		# Append the predicted values into a list
		Predicted = []
		for i in predicted_prices:
			Predicted.append(i[0])
			#Predicted.append(round(i[0], 3))


		
		# Append the close values to the list
		close = []
		for i in self.priceVolTargetScaledData:
			close.append(i[0])



		# Create a dataframe based on the dates in the individual stock data
		df_predicted = self.priceVolTargetData[['Date']]

		# Add the close values to the dataframe
		df_predicted['Close'] = close

		# Add the predicted values to the dataframe
		df_predicted['predictions'] = Predicted
		self.close=close[1:]
		self.ridgePredicted=df_predicted
		self.ridge_acc=self.rss(np.array(self.ridgePredicted['predictions']),np.array(close))
		return self.ridgePredicted

	def get_ridge_acc(self):
		return self.ridge_acc

	def get_ridgePredicted(self):
		return self.ridgePredicted

	def print_ridge_regression_result(self,stock_name):
		df_predicted=self.ridge_regression_training(stock_name)
		# Plot the results
		self.interactive_plot(df_predicted,"Ridge", "Original Vs Prediction ("+stock_name+" Ridge)")

