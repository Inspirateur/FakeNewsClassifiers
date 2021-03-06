from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from Logic.Datasets.dataset import Dataset
import Logic.Datasets.dataloader as dataloader


class Classifier:
	def __init__(self, data: str):
		self.d: Dataset = dataloader.get(data)
		print(self.d.detail())

	def evaluate(self):
		"""
		Evaluate the model once, doesn't train it
		"""
		# define the confusion matrices
		c = self.d.classes
		conf_mat = np.zeros(shape=(len(c), len(c)), dtype=np.float)
		preds = self.predict(self.d.test.X)
		# it is possible that some prediction get None score with some classifier
		# we don't count them in the accuracy score
		preds = np.ma.masked_invalid(preds)
		y = np.ma.masked_where(preds.mask, self.d.test.y)
		for j in range(len(preds)):
			conf_mat[preds[j], y[j]] += 1
		self.display_confmat(conf_mat)

	def k_evaluate(self, k=10):
		"""
		Use k-fold validation to evaluate the model, train it each time
		"""
		# define the confusion matrices
		c = self.d.classes
		conf_mat = np.zeros(shape=(k, len(c), len(c)), dtype=np.float)
		# _train and predict k times
		for i in range(k):
			self.d.shuffle()
			self.train()
			preds = self.predict(self.d.test.X)
			# it is possible that some prediction get None score with some classifier
			# we don't count them in the accuracy score
			preds = np.ma.masked_invalid(preds)
			y = np.ma.masked_where(preds.mask, self.d.test.y)
			for j in range(len(preds)):
				conf_mat[i, preds[j], y[j]] += 1
		# average the unormalized confmat
		self.display_confmat(conf_mat.mean(axis=0))

	def display_confmat(self, conf_mat):
		# compute the accuracy
		acc = np.sum(conf_mat.diagonal())/np.sum(conf_mat)
		print(f"acc={acc:.1%}")
		# normalize the confmat along the prediction axis
		conf_mat /= conf_mat.sum(axis=0)
		plt.imshow(conf_mat)
		plt.colorbar()
		tickvals = list(range(len(self.d.classes)))
		ticklabs = list(self.d.classes.keys())
		plt.xticks(tickvals, ticklabs, rotation=45, ha="right")
		plt.yticks(tickvals, ticklabs)
		plt.clim(0, 1)
		plt.tight_layout()
		plt.show()

	def train(self):
		"""
		Train the model, implement this unless you don't need the k-fold validation
		to compute the accuracy
		"""
		pass

	def predict(self, inputs: np.ndarray) -> np.ndarray:
		"""
		Return the predictions for the batch of inputs provided
		"""
		raise NotImplementedError

	def save(self):
		"""
		Must save the model on the disk
		"""
		pass

	def load(self):
		"""
		Must load the model from the disk
		:raises: FileNotFoundError if the save file is not found
		(also raises that by default instead of NotImplementedError for convenience)
		"""
		raise FileNotFoundError
