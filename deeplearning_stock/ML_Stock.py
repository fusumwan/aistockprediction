
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
from sklearn.metrics import mean_squared_error

'''
///-- =============================================
///-- Author: Sum Wan,FU
///-- Create date: 2021-10-21
///-- Student ID: a1714470
///-- Course:Applied Machine Learning (merged 4416_4816_7416)
///-- This class include some function to do standardization approach for data compatibility.
///-- Because we need to identifying and correcting mistakes or errors in the data.
///-- If the results of referencing a numeric field that contains non-numeric, or otherwise invalid data, are undefined. Then such conditions may be detected, and give an error at run-time.
///-- Hence, we have to check 2160 stock data records with 10 attributes. 
///-- =============================================
'''


class ML_Stock(AI.MLbase):
	# instance attribute
	GraphicControl=MLG.MLGraphic(False)
	def __init__(self, dx, isShowFigure, filename1,filename2):
		super().__init__(dx, isShowFigure, filename1,filename2)
		self.GraphicControl=MLG.MLGraphic(isShowFigure)
		#print(stock_price_df)
	def plt_save_figure(self,plt,directory,filename):
		self.GraphicControl.plt_save_figure(plt,directory,filename)


	def fig_save_figure(self,fig,directory,filename):
		self.GraphicControl.fig_save_figure(fig,directory,filename)

	# Define a data plotting function
	def show_plot2(self,data, directory, title):
		self.GraphicControl.show_plot2(data, directory, title)


	# Function to plot interactive plots using Plotly Express
	def interactive_plot(self,df,directory, title):
		self.GraphicControl.interactive_plot(df,directory, title)
 
	def show_plot (self,df,directory,title):
		self.GraphicControl.show_plot(df,directory,title)

	def show_plot_ex (self,title):
		self.GraphicControl.show_plot(self.stockPriceData,title)


	def log(self,*argv):
			for arg in argv:
				print (arg)

	def get_stockPriceData(self):
		return self.stockPriceData

	def check_null_stock_price_data(self):
		# Check if Null values exist in stock prices data
		return self.stockPriceData.isnull().sum()

	def check_null_stock_volume_data(self):
		# Check if Null values exist in stocks volume data
		return self.stockVolData.isnull().sum()

	def get_stock_prices_dataframe_info(self):
		# Get stock prices dataframe info
		return self.stockPriceData.info()

	def get_stock_volume_dataframe_info(self):
		# Get stock volume dataframe info
		return self.stockVolData.info()

	def get_stock_volume_dataframe_describe(self):
		#Get stock volume dataframe describe
		return self.stockVolData.describe() 

	# Function to normalize stock prices based on their initial price
	def normalize(self,df):
		x = df.copy()
		for i in x.columns[1:]:
			x[i] = x[i]/x[i][0]
		return x

	
	
	# Function to concatenate the date, stock price, and volume in one dataframe
	def individual_stock(self,price_df, vol_df, name):
		return pd.DataFrame({'Date': price_df['Date'], 'Close': price_df[name], 'Volume': vol_df[name]})



	# Function to return the input/output (target) data for AI/ML Model
	# Note that our goal is to predict the future stock price 
	# Target stock price today will be tomorrow's price 
	def target_label(self,data):
		
		# 1 day window 
		n = 1

		# Create a column containing the prices for the next 1 days
		data['Target'] = data[['Close']].shift(-n)
		
		# return the new dataset 
		return data

	def get_individual_stock_prices_volume(self,stock_name):
		# Test the functions and get individual stock prices and volumes for 'AAPL'
		self.priceVolData = self.individual_stock(self.stockPriceData, self.stockVolData, stock_name)
		return self.priceVolData

	def get_individual_stock_prices_volume_target(self):
		# Test the functions and get target label
		self.priceVolTargetData = self.target_label(self.priceVolData)
		return self.priceVolTargetData

	def remove_price_volume_null_value(self):
		# Remove the last row as it will be a null value
		self.priceVolTargetData = self.priceVolTargetData[:-1]
		return self.priceVolTargetData

	def scale_the_date(self):
		# Scale the data
		sc = MinMaxScaler(feature_range = (0, 1))
		self.priceVolTargetScaledData = sc.fit_transform(self.priceVolTargetData.drop(columns = ['Date']))
		return self.priceVolTargetScaledData

	def creating_feature_and_target(self):
		# Creating Feature and Target
		self.X = self.priceVolTargetScaledData[:,:2]
		self.Y = self.priceVolTargetScaledData[:,2:]
		self.X = np.asarray(self.X)
		self.Y = np.asarray(self.Y)

	def spliting_data(self):
		# Spliting the data this way, since order is important in time-series
		# Note that we did not use train test split with it's default settings since it shuffles the data
		split = int(0.65 * len(self.X))
		self.XTrainData = self.X[:split]
		self.YTrainData = self.Y[:split]
		self.XTrainData2 = self.X[:split]
		self.YTrainData2 = self.Y[:split]
		self.XTestData = self.X[split:]
		self.YTestData = self.Y[split:]
		self.XTestData2 = self.X[split:]
		self.YTestData2 = self.Y[split:]

	def stock_daily_return (self,stock_data):
		d_ret = stock_data.copy()
		for i in stock_data.columns[1:]:
				for j in range (1,len(stock_data)):
					day_price=stock_data[i][j]
					pre_day_price=stock_data[i][j-1]
					#print(str(i)+"=> day_price:"+str(day_price) +"   pre_day_price:"+str(pre_day_price)+" pre_day_price:"+str(pre_day_price))
					d_ret[i][j] = ((day_price - pre_day_price) / pre_day_price) * 100
				d_ret[i][0] = 0
		return d_ret
	
	def daily_return (self,df):
		df_daily_retun = df.copy()
		for i in df.columns[1:]:
				for j in range (1,len(df)):
					day_price=df[i][j]
					pre_day_price=df[i][j-1]

					callable(day_price)
					callable(pre_day_price)
					try:
							float(day_price)
					except ValueError:
							print("day_price Not a float")

					try:
							float(pre_day_price)
					except ValueError:
							print("pre_day_price Not a float")
					
					df_daily_retun[i][j] = ((df[i][j] - df[i][j-1]) / df[i][j-1]) * 100
						
						
				df_daily_retun[i][0] = 0
				
		return df_daily_retun

	def rss(self,y_pred,y_true):
		u=((y_true - y_pred) ** 2).sum()
		v=((y_true - y_true.mean()) ** 2).sum()
		r=(1-u/v)
		return r

	def set_ridgePredicted(self,rp):
		self.ridgePredicted=rp

	def set_lstmPredicted(self,lstm):
		self.lstmPredicted=lstm

	def calc_averagePredicted(self,p):
		self.averagePredicted = self.lstmPredicted
		self.averagePredicted['predictions']=(np.array(self.ridgePredicted['predictions'])*(1-p)+np.array(self.lstmPredicted['predictions'])*p)

	def get_averagePredicted(self,p):
		self.calc_averagePredicted(p)
		return self.averagePredicted

	def get_close(self):
		return self.close

	def print_result(self,stock_name):
		self.interactive_plot(self.averagePredicted,"Average", "Original Vs Prediction("+stock_name+")")

	def print_original_result(self):
		self.show_plot (self.stockPriceData,'Original', 'Stock Prices 2012-01-12 to 2020-08-11 (Without Normilized)')
		self.show_plot (self.normalize(self.stockPriceData),'Original', 'Stock Prices 2012-01-12 to 2020-08-11 (Normilized)')
		stock_daily_return_df = self.stock_daily_return(self.stockPriceData)
		#self.show_plot(stock_daily_return_df,'Original', "Historical stock price daily return")
		#plot interactive chart for stocks data by Plotly express
		self.interactive_plot(self.stockPriceData,'Original', 'Stock Prices 2012-01-12 to 2020-08-11 Uing Plotly (Without Normilized)')
		self.interactive_plot(self.normalize(self.stockPriceData),'Original', 'Stock Prices 2012-01-12 to 2020-08-11 Using Plotly(Normilized)')

		
	def print_stock_original_result(self,stock_name):
		self.get_individual_stock_prices_volume(stock_name)
		self.get_individual_stock_prices_volume_target()
		self.remove_price_volume_null_value()
		self.scale_the_date()
		self.creating_feature_and_target()
		self.spliting_data()
		# plot interactive chart for stocks data (Close(red) vs Volume(green))
		self.show_plot2(self.XTrainData,'Original', "Training Data Close Vs Volume ("+stock_name+" Normilized)")
		self.show_plot2(self.XTestData,'Original', "Testing Data Close Vs Volume ("+stock_name+" Normilized)")


	def categorical_accuracy(self,y_true, y_pred):
		return K.cast(K.equal(K.argmax(y_true, axis=-1),
							  K.argmax(y_pred, axis=-1)),
					  K.floatx())


	def sparse_categorical_accuracy(self,y_true, y_pred):
		return K.cast(K.equal(K.max(y_true, axis=-1),
							  K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
					  K.floatx())



