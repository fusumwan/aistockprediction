
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
///-- This class include LSTM regression function
///-- =============================================
'''

class LSTMAI(ML.ML_Stock):
	def __init__(self, dx, isShowFigure, filename1,filename2):
		super().__init__(dx, isShowFigure, filename1,filename2)

	def lstm_training(self,stock_name):
		
		self.get_individual_stock_prices_volume(stock_name)
		self.get_individual_stock_prices_volume_target()
		self.remove_price_volume_null_value()
		self.scale_the_date()

		self.creating_feature_and_target()
		self.spliting_data()

		# Get the close and volume data as training data (Input)
		training_data = self.priceVolData.iloc[:, 1:3].values

		# Normalize the data
		sc = MinMaxScaler(feature_range = (0, 1))
		self.trainingSetScaled = sc.fit_transform(training_data)

		# Create the training and testing data, training data contains present day and previous day values
		self.X = []
		self.Y = []
		for i in range(1, len(self.priceVolData)):
				self.X.append(self.trainingSetScaled [i-1:i, 0])
				self.Y.append(self.trainingSetScaled [i, 0])

		# Convert the data into array format
		self.X = np.asarray(self.X)
		self.Y = np.asarray(self.Y)

		# Split the data
		split = int(0.7 * len(self.X))
		self.XTrainData = self.X[:split]
		self.YTrainData = self.Y[:split]
		self.XTestData = self.X[split:]
		self.YTestData = self.Y[split:]

		# Reshape the 1D arrays to 3D arrays to feed in the model
		self.XTrainData = np.reshape(self.XTrainData, (self.XTrainData.shape[0], self.XTrainData.shape[1], 1))
		self.XTestData = np.reshape(self.XTestData, (self.XTestData.shape[0], self.XTestData.shape[1], 1))

		# Create the model
		deepnet=keras
		input_layer = deepnet.layers.Input(shape=(self.XTrainData.shape[1], self.XTrainData.shape[2]))
		xlayer = deepnet.layers.LSTM(150, return_sequences= True)(input_layer)
		xlayer = deepnet.layers.Dropout(0.3)(xlayer)
		xlayer = deepnet.layers.LSTM(150, return_sequences=True)(xlayer)
		xlayer = deepnet.layers.Dropout(0.3)(xlayer)
		xlayer = deepnet.layers.LSTM(150)(xlayer)
		output_layer = deepnet.layers.Dense(1, activation='linear')(xlayer)

		lstm_model = deepnet.Model(inputs=input_layer, outputs=output_layer)
		#lstm_model.compile(optimizer='sgd', loss="mse", metrics=[keras.metrics.SparseCategoricalAccuracy()])
		lstm_model.compile(optimizer='adam', loss="mse")
		lstm_model.summary()

		# Trai the model
		lstm_history = lstm_model.fit(
				self.XTrainData, self.YTrainData,
				epochs = 20,
				batch_size = 32,
				validation_split = 0.2
		)

		# Make prediction
		lstmPredicted = lstm_model.predict(self.X)

		# Append the predicted values to the list
		lstm_test_predicted = []
		for i in lstmPredicted:
			#print(i[0])
			lstm_test_predicted.append(i[0])



		lstm_df_predicted = self.priceVolData[1:][['Date']]
		lstm_df_predicted['predictions'] = lstm_test_predicted
		
		# Plot the data
		close = []
		for i in self.trainingSetScaled:
			close.append(i[0])

		lstm_df_predicted['Close'] = close[1:]
		self.lstmPredicted=lstm_df_predicted
		self.close=close[1:]



		self.lstm_acc=self.rss(np.array(self.lstmPredicted['predictions'] ),np.array(self.lstmPredicted['Close']))
		
		return self.lstmPredicted

	def get_lstm_acc(self):
		return self.lstm_acc

	def get_lstmPredicted(self):
		return self.lstmPredicted

	def print_lstm_result(self,stock_name):
		df_predicted=self.lstm_training(stock_name)
		# Plot the results
		self.interactive_plot(df_predicted,"LSTM", "Original Vs Prediction ("+stock_name+" LSTM)")


