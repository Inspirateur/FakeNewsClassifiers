from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from Logic.preprocessing import Vectorizer
from Logic.Datasets.dataset import Dataset
import Logic.Datasets.dataloader as dataloader


class Classifier:
	def __init__(self, data: str, vectorizer: Vectorizer = None):
		self.d: Dataset = dataloader.get(data)
		print(self.d.detail())
		self.vec = vectorizer

	def evaluate(self, k=10) -> np.ndarray:
		"""
		Use _train and _predict to perform a k-fold validation and compute the confusion matrix
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
		conf_mat = conf_mat.mean(axis=0)
		# compute the accuracy
		acc = np.sum(conf_mat.diagonal())/np.sum(conf_mat)
		print(f"acc={acc:.1%}")
		# normalize the confmat along the prediction axis
		conf_mat /= conf_mat.sum(axis=0)
		plt.imshow(conf_mat)
		plt.colorbar()
		tickvals = list(range(len(self.d.classes)))
		ticklabs = self.d.classes.tolist()
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

	def analyze(self, query: str) -> Tuple[float, str]:
		"""
		Given a single input with the tokenized version and a vocab, return a score and an analysis
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
