from typing import Tuple
import numpy as np


class Classifier:
	def train(self, inputs: np.ndarray, labels: np.ndarray, k=10) -> float:
		"""
		Use _train and _predict to perform a k-fold validation and compute the accuracy
		"""
		# define the train and validation split
		split = int(inputs.shape[0]*.95)
		accs = np.empty(shape=(k,), dtype=np.float)
		# _train and predict k times
		for i in range(k):
			p = np.random.permutation(inputs.shape[0])
			_inputs = inputs[p]
			_labels = labels[p]
			self._train(_inputs[:split], _labels[:split])
			accs[i] = (self.predict(_inputs[split:]) == _labels[split:]).mean()
		# train a last time on the whole training set
		self._train(inputs, labels)
		# return the average accuracy
		return accs.mean()

	def _train(self, inputs: np.ndarray, labels: np.ndarray):
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

	def analyze(self, x, tokens) -> Tuple[float, str]:
		"""
		Given a single input with the tokenized version, return a score and an analysis
		"""
		raise NotImplementedError

	def save(self):
		"""
		Must save the model on the disk
		"""
		raise NotImplementedError

	def load(self):
		"""
		Must load the model from the disk
		:raises: FileNotFoundError if the save file is not found
		"""
		raise NotImplementedError
