import pickle
from typing import Tuple
import numpy as np
from Logic.Classifiers.classifier import Classifier


def sigmoid(x):
	return 1./(1.+np.exp(-x))


class WikiLinksClassifier(Classifier):
	def __init__(self):
		print("Loading Wikipedia links data...", end=' ', flush=True)
		with open("Logic/ExternalData/infolinks.pickle", "rb") as flinks:
			self.links = pickle.load(flinks, fix_imports=False)
		with open("Logic/ExternalData/redirects.pickle", "rb") as fredirs:
			self.redirs = pickle.load(fredirs, fix_imports=False)
		print("Done")

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

	def analyze(self, x, tokens, _) -> Tuple[float, str]:
		x = x[0]
		token_set = set()
		known_tokens = []
		for token in x:
			token = token.lower()
			if token not in token_set:
				if token in self.redirs:
					token = self.redirs[token]
				if token in self.links:
					token_set.add(token)
					known_tokens.append(token)
		if len(known_tokens) <= 1:
			return None, ""
		known_tokens = known_tokens[0:2]
		path = self.shortest_path(known_tokens[0], known_tokens[1])
		if path:
			return sigmoid(len(path)*-1+3), f"<p>{' -> '.join(reversed(path))}</p>"
		else:
			return None, ""

	def save(self):
		pass

	def load(self):
		pass
