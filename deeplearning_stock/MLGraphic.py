
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
'''
///-- =============================================
///-- Author: Sum Wan,FU
///-- Create date: 2021-10-21
///-- Student ID: a1714470
///-- Course:Applied Machine Learning (merged 4416_4816_7416)
///-- This class include the plotting function for plotting the graphics.
///-- =============================================
'''

class MLGraphic:
	isShowFigure = False
	def __init__(self,isShowFigure):
		self.isShowFigure=isShowFigure

	def plt_save_figure(self,plt,directory,filename):
		script_dir = os.path.dirname(__file__)
		results_dir = os.path.join(script_dir, directory+'/')
		if not os.path.isdir(results_dir):
			os.makedirs(results_dir)
		plt.savefig(results_dir + filename+".png")

	def fig_save_figure(self,fig,directory,filename):
		script_dir = os.path.dirname(__file__)
		results_dir = os.path.join(script_dir, directory+'/')
		if not os.path.isdir(results_dir):
			os.makedirs(results_dir)
		pio.write_image(fig,results_dir + filename+".png", format='png')


	# Define a data plotting function
	def show_plot2(self,data, directory, title):
		plt.figure(figsize = (13, 5))
		plt.plot(data, linewidth = 3)
		
		for ax in plt.gcf().axes:
			ax.get_lines()[0].set_color("green")
			ax.get_lines()[1].set_color("red")
		
		plt.title(title)
		plt.grid()
		self.plt_save_figure(plt,directory,title)
		if self.isShowFigure==True:
			plt.show()


	# Function to plot interactive plots using Plotly Express
	def interactive_plot(self,df,directory, title):
		fig = px.line(title = title)
		for i in df.columns[1:]:
			fig.add_scatter(x = df['Date'], y = df[i], name = i)
		self.fig_save_figure(fig,directory,title)
		if self.isShowFigure==True:
			fig.show()
 
	def show_plot (self,df,directory,title):
		df.plot (x = 'Date', figsize = (15,8), title = title, linewidth = 2)
		plt.grid()
		self.plt_save_figure(plt,directory,title)
		if self.isShowFigure==True:
			plt.show()