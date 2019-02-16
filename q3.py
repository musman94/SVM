#from __future__ import division
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

class q3(object):
	def __init__(self):
		self.cancer_data = pd.read_csv("./UCI_Breast_Cancer.csv", header = None).values
		self.data = self.cancer_data[:, 1: 10]
		self.labels = self.cancer_data[:, 10:]

		self.X_train, self.X_test, self.y_train, self.y_test = self.data[:500, :], self.data[500:, :], self.labels[:500, :].ravel(), self.labels[500:, :].ravel()

		self.c_vals = np.logspace(-10, 10, 21)
		self.gamma_vals = 2 * np.logspace(-10, 10, 21)

		self.k_fold_accuracies = []

		self.C = 1
		self.gamma = 1

	def compare(self, predictions, y, test = False):
		tp, tn, fp, fn = 0, 0, 0, 0
		for x in xrange(len(y)):
			if predictions[x] == 4 and y[x] == 4:
				tp += 1
			elif predictions[x] == 2 and y[x] == 2:
				tn += 1
			elif predictions[x] == 4 and y[x] == 2:
				fp += 1
			else:
				fn += 1

		if test:
			print "True Positives: {}".format(tp)
			print "True Negatives: {}".format(tn)
			print "False Positives: {}".format(fp)
			print "False Negatives: {}".format(fn)

		return  float((tp + tn)) / float((tp + tn + fp + fn)) 

	def plot(self, gamma = False):
		plt.xscale('log')
		if gamma:
			plt.plot(self.gamma_vals, self.k_fold_accuracies)
			plt.xlabel("gamma")
		else:
			plt.plot(self.c_vals, self.k_fold_accuracies)
			plt.xlabel("C")
		plt.ylabel("Accuracy")
		plt.show()


	def k_fold(self, X_train, y_train, C, div, gamma = None):
		overall_accuracy = 0
		if gamma:
			model = svm.SVC(C = C, kernel = 'rbf', gamma = gamma, random_state = 1000)
		else:
			model = svm.LinearSVC(C = C, random_state = 1000)
		for x in xrange(10):
			start = x * div
			end = x * div + div
			X_valid = X_train[start : end, :]
			y_valid = y_train[start : end]
			X_inner_train = np.concatenate([X_train[:start, :], X_train[end:, :]])
			y_inner_train = np.concatenate([y_train[:start], y_train[end:]])
			model.fit(X_inner_train, y_inner_train)
			predictions = model.predict(X_valid)
			current_accuarcy = self.compare(predictions, y_valid)
			overall_accuracy += current_accuarcy
		return overall_accuracy

	def q3_3(self):
		max_accuracy = 0
		div = len(self.X_train) / 10
		for c in self.c_vals:
			k_fold_accuarcy = self.k_fold(self.X_train, self.y_train, c, div, None) / 10.0
			self.k_fold_accuracies.append(k_fold_accuarcy)
			if k_fold_accuarcy > max_accuracy:
				self.C = c
				max_accuracy = k_fold_accuarcy

		###Testing###
		model = svm.LinearSVC(C = self.C, random_state = 1000)
		model.fit(self.X_train, self.y_train)
		predictions = model.predict(self.X_test)
		test_accuarcy = self.compare(predictions, self.y_test, True)
		print "Best C value: {}".format(self.C)
		print "Accuracy: {:0.2f}".format(test_accuarcy)

		###plot###
		self.plot()

	def q3_4(self):
		max_accuracy = 0
		self.k_fold_accuracies = []
		div = len(self.X_train) / 10
		for gamma in self.gamma_vals:
			k_fold_accuarcy = self.k_fold(self.X_train, self.y_train, self.C, div, gamma) / 10.0
			self.k_fold_accuracies.append(k_fold_accuarcy)
			if k_fold_accuarcy > max_accuracy:
				self.gamma = gamma
				max_accuracy = k_fold_accuarcy
		###Testing###
		model = svm.SVC(C = self.C, kernel = 'rbf', gamma = self.gamma, random_state = 1000)
		model.fit(self.X_train, self.y_train)
		predictions = model.predict(self.X_test)
		test_accuarcy = self.compare(predictions, self.y_test, True)
		print "Best gamma value: {}".format(self.gamma)
		print "Accuracy: {:0.2f}".format(test_accuarcy)

		###plot###
		self.plot(True)



q3 = q3()
q3.q3_3()
q3.q3_4()
