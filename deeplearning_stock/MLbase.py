
import numpy as np
import pandas as pd

'''
///-- =============================================
///-- Author: Sum Wan,FU
///-- Create date: 2021-10-21
///-- Student ID: a1714470
///-- Course:Applied Machine Learning (merged 4416_4816_7416)
///-- This is the base class that includes common parameters.
///-- =============================================
'''

class MLbase:
	# class attribute
	X=[]
	Y=[]
	dx = 0
	isShowFigure = False
	StockName=[]
	XTrainData=[]
	YTrainData=[]
	XTrainData2=[]
	YTrainData2=[]
	XTestData=[]
	YTestData=[]
	stockPriceData = []
	stockVolData = []
	priceVolData = []
	priceVolTargetData=[]
	priceVolTargetScaledData=[]
	trainingSetScaled=[]
	ridgePredicted=[]
	lstmPredicted=[]
	averagePredicted=[]
	
	lstm_loss=0.0
	lstm_acc=0.0
	ridge_acc=0.0
	def __init__(self, dx, isShowFigure, filename1,filename2):
		self.dx = dx
		self.isShowFigure=isShowFigure
		# Read stock prices data
		self.stockPriceData = pd.read_csv(filename1)

		# Read the stocks volume data
		# Stock Volume is counted as the total number of shares that are actually traded (bought and sold) during the trading day or specified set period of time. It is a measure of the total turnover of shares.
		self.stockVolData = pd.read_csv(filename2)

