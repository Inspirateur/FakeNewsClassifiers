from typing import Tuple
import numpy as np
from Logic.Classifiers.classifier import Classifier
from Logic.preprocessing import Vectorizer


class DumbClassifier(Classifier):
	def __init__(self, data: str, vectorizer: Vectorizer = None, best_label=None):
		Classifier.__init__(self, data, vectorizer)
		self.best_label = best_label

	def train(self):
		values, counts = np.unique(self.d.train.y, return_counts=True)
		i = np.argmax(counts)
		self.best_label = i

	def predict(self, inputs: np.ndarray) -> np.ndarray:
		return np.full(shape=inputs.shape, fill_value=self.best_label)

	def analyze(self, x) -> Tuple[float, str]:
		return np.random.random(), "<p>Just because</p>"
