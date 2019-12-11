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
		return np.random.random(size=(inputs.shape[0],))

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

	def analyze(self, x, tokens, vocab) -> Tuple[float, str]:
		known_tokens = {}
		for token in tokens:
			_token = token
			if token in self.redirs:
				token = self.redirs[token]
			if token in self.links:
				known_tokens[token] = max(
					known_tokens[token] if token in known_tokens else 0,
					x[0, vocab[token]] if token in vocab else 0,
					x[0, vocab[_token]] if _token in vocab else 0
				)
		if len(known_tokens) <= 1:
			return None, ""
		source, target = sorted(
			sorted(
				enumerate(known_tokens.items()), key=lambda iws: iws[1][1], reverse=True
			)[0:2],  # selects the 2 words with the most significance (according to the weighting of x)
			key=lambda iws: iws[0]  # sort them by order of appearance in the sentence
		)
		source = source[1][0]
		target = target[1][0]
		path = self.shortest_path(source, target)
		if path:
			return sigmoid(len(path)*-1+3), f"<p>The path between <b>{source}</b> and <b>{target}</b> is:</p>" \
											f"<p>{' -> '.join(reversed([target]+path))}</p>"
		else:
			return None, ""
