
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
'''
///-- =============================================
///-- Author: Sum Wan,FU
///-- Create date: 2021-10-21
///-- Student ID: a1714470
///-- Course:Applied Machine Learning (merged 4416_4816_7416)
///-- This class is used for calculating accuracy.
///-- =============================================
'''

class funcInterFace:
	close=[]
	ridgePredicted = []
	lstmPredicted = []
	df=0.0
	weightList1=0
	weightList2=0
	ridge_acc=0
	lstm_acc=0
	def __init__(self, c,rp,lp,ridge_acc,lstm_acc):
		self.close=c
		self.ridgePredicted=rp
		self.lstmPredicted=lp
		self.ridge_acc=ridge_acc
		self.lstm_acc=lstm_acc
	def rss(self,y_pred,y_true):
		u=((y_true - y_pred) ** 2).sum()
		v=((y_true - y_true.mean()) ** 2).sum()
		r=(1-u/v)
		return r
	def start_x(self):
		if self.ridge_acc > self.lstm_acc:
			return 0
		else:
			return 1
	def compare_acc(self,x,acc):
		if self.ridge_acc > acc:
			return 0
		if self.lstm_acc > acc:
			return 1
		return x
	def cal(self,x):
		weightList1=(1-x)
		weightList2=x
		if x==0:
			r=self.ridgePredicted
		elif x==1:
			r=self.lstmPredicted
		else:
			r=(self.ridgePredicted*weightList1+self.lstmPredicted*weightList2)

		return self.rss(r,self.close)
	def get_ridgePredicted(self):
		return self.ridgePredicted
	def get_lstmPredicted(self):
		return self.lstmPredicted
	def get_close(self):
		return self.close
	def get_avgAcc(self,x):
		r=(self.ridgePredicted*(1-x)+self.lstmPredicted*x)
		return mean_squared_error(r,self.close)
	def get_ridgeAcc(self):
		return mean_squared_error(self.ridgePredicted,self.close)
	def get_lstmAcc(self):
		return mean_squared_error(self.lstmPredicted,self.close)