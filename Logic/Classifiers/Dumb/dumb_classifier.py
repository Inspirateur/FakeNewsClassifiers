from typing import Tuple
import numpy as np
from Logic.Classifiers.classifier import Classifier


class DumbClassifier(Classifier):
	def __init__(self, best_label=None):
		self.best_label = best_label

	def _train(self, inputs: np.ndarray, labels: np.ndarray):
		values, counts = np.unique(labels, return_counts=True)
		self.best_label = values[np.argmax(counts)]

	def predict(self, inputs: np.ndarray) -> np.ndarray:
		return np.full(shape=inputs.shape, fill_value=self.best_label)

	def analyze(self, x, tokens, vocab) -> Tuple[float, str]:
		return np.random.random(), "<p>Just because</p>"
