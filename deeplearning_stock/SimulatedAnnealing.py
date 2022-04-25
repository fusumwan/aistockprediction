import numpy as np
import pandas as pd
import random
'''
///-- =============================================
///-- Author: Sum Wan,FU
///-- Create date: 2021-10-21
///-- Student ID: a1714470
///-- Course:Applied Machine Learning (merged 4416_4816_7416)
///-- This is class is used for Simulated Annealing algorithm.
///-- For various reasons, after studying the the simulated annealing/hill climbing algorithm. 
///-- Such algorithm is chosen to find the best weights for applying the effective averaging method. 
///-- The simulated annealing/hill climbing algorithm is used as a simple greedy search algorithm. 
///-- Because we need a algorithm to select an optimal solution from the adjacent solution space of the current solution as the current solution each time until a local optimal solution is reached.
///-- =============================================
'''
class SimulatedAnnealing:
	# class attribute
	dx = 0
	acc = 0
	sc_acc=0
	percentage = 0
	close=[]
	averagePredicted = []
	ridgePredicted = []
	lstmPredicted = []
	# instance attribute
	def __init__(self, dx):
		self.dx = dx

	def hillClimbingCalMax(self,f):
		i = 1
		acc=0.0
		x=f.start_x()
		greaterThan = float(self.dx)
		digits = int(20)
		divide=1000000
		while i <=divide :
			lessThan = 0.9-(i/divide)*0.9
			if lessThan<greaterThan:
				greaterThan = lessThan
			if lessThan<=0:
				lessThan = 0
				greaterThan = 0
				rounded_number=0
			else:
				rounded_number = round(random.uniform(greaterThan, lessThan), digits)
			if rounded_number<0:
				rounded_number=0.0

			if(f.cal(x+self.dx+rounded_number)>=f.cal(x)):
				x=x+self.dx+rounded_number
			if(f.cal(x-self.dx-rounded_number)>=f.cal(x)):
				x=x-self.dx-rounded_number
			acc=f.cal(x)
			self.print_acc(divide, i,f,x)
			i+=1;
			self.acc=acc
		x=f.compare_acc(x,self.acc)
		self.acc=f.cal(x)
		self.print_acc(divide, divide,f,x)
		self.percentage=x
		self.close=f.get_close()
		return self.percentage

	def hillClimbingCalMin(self,f):
		i = 1
		acc=0.0
		x=f.start_x()
		greaterThan = float(self.dx)
		sub = float(self.dx)*0.01
		digits = int(20)
		divide=1000000
		while i <=divide :
			
			lessThan = 0.9-(i/divide)*0.9
			if lessThan<greaterThan:
				greaterThan = lessThan
			if lessThan<=0:
				lessThan = 0
				greaterThan = 0
				rounded_number=0
			else:
				rounded_number = round(random.uniform(greaterThan, lessThan), digits)
			if rounded_number<0:
				rounded_number=0.0
			
			if(f.cal(x+self.dx+rounded_number)<=f.cal(x)):
				x=x+self.dx+rounded_number
			if(f.cal(x-self.dx-rounded_number)<=f.cal(x)):
				x=x-self.dx-rounded_number
			acc=f.cal(x)
			self.print_acc(divide, i,f,x)
			i+=1;
			self.acc=acc
		x=f.compare_acc(x,f.cal(x))
		self.acc=f.cal(x)
		self.print_acc(divide,dividef,x)
		self.percentage=x
		self.close=f.get_close()
		return self.percentage

	def log(self,*argv):
			for arg in argv:
				print (arg)
	def print_acc(self,max_loop,i,f,x):
		process= (i/max_loop) * 100
		self.log("Process:" +str(str(round(process, 2)))+"%"+" Best Finding:f.cal(" +str(x)+")=" + str(f.cal(x)))

	def get_averagePredicted(self):
		return self.averagePredicted

	def getPercentage(self):
		return self.percentage

	def getAccuracy(self):
		return self.acc

	def getClose(self):
		return self.close
