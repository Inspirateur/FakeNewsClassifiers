import numpy as np
from Logic.Classifiers.classifier import Classifier


class DumbClassifier(Classifier):
	def __init__(self, data: str, best_label=None):
		Classifier.__init__(self, data)
		self.best_label = best_label

	def train(self):
		values, counts = np.unique(self.d.train.y, return_counts=True)
		i = np.argmax(counts)
		self.best_label = i

	def predict(self, inputs: np.ndarray) -> np.ndarray:
		return np.full(shape=inputs.shape, fill_value=self.best_label)
