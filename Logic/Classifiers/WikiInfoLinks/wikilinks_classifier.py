import json
from typing import Tuple
import numpy as np
from Logic.Classifiers.classifier import Classifier
from Logic.preprocessing import Vectorizer


def sigmoid(x):
	return 1./1.+np.exp(-x)


class WikiLinksClassifier(Classifier):
	def __init__(self):
		with open("Logic/Datasets/Wikipedia-Infolinks/infolinks.json", "r") as flinks:
			self.links = json.load(flinks)

	def _train(self, inputs: np.ndarray, labels: np.ndarray):
		pass

	def predict(self, inputs: np.ndarray) -> np.ndarray:
		return np.random.random(size=inputs.shape)

	def shortest_path(self, source, target, steps=5):
		for node in self.links[source]:
			if node == target:
				return [source]
			if steps > 0 and node in self.links:
				path = self.shortest_path(node, target, steps - 1)
				if path:
					path.append(source)
					return path
		return None

	def analyze(self, x, tokens) -> Tuple[float, str]:
		x = x[0]
		known_tokens = set()
		for token in x:
			if token.lower() in self.links:
				known_tokens.add(token.lower())
		known_tokens = list(known_tokens)
		if len(known_tokens) <= 1:
			return None, ""
		known_tokens = known_tokens[0:2]
		path = self.shortest_path(known_tokens[0], known_tokens[1])
		if path:
			return sigmoid(len(path)*-1+3), path
		else:
			return None, ""

	def save(self):
		pass

	def load(self):
		pass
