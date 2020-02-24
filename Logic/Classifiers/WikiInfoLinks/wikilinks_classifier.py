from collections import defaultdict
import pickle
from time import time
from typing import Tuple
import numpy as np
from Logic.Classifiers.classifier import Classifier
from Logic.preprocessing import Vectorizer


def get_sentences(x):
	start = 0
	for i, token in enumerate(x):
		if token == '.':
			yield x[start:i]
			start = i + 1
	if start < len(x):
		yield x[start:]


class WikiLinksClassifier(Classifier):
	weights: dict
	alpha: float

	def __init__(self, data: str, vectorizer: Vectorizer = None, alpha=3):
		Classifier.__init__(self, data, vectorizer)
		self.alpha = alpha
		self.cache: dict = {}
		print("Loading Wikipedia links data...", end=' ', flush=True)
		start = time()
		with open("Logic/ExternalData/infolinks.pickle", "rb") as flinks:
			self.links = pickle.load(flinks, fix_imports=False)
		with open("Logic/ExternalData/redirects.pickle", "rb") as fredirs:
			self.redirs = pickle.load(fredirs, fix_imports=False)
		print(f"Done in {time()-start:.1f} sec")

	def shortest_path(self, source, target, steps=5):
		# TODO: Passer ca en BFS (pour retourner le plus court)
		if (source, target) in self.cache:
			return self.cache[(source, target)]
		for node in self.links[source]:
			if node == target:
				return [source]
			if steps > 0 and node in self.links:
				path = self.shortest_path(node, target, steps - 1)
				if path:
					path.append(source)
					self.cache[(source, target)] = path.copy()
					return path
		return None

	def get_source_target(self, sentence):
		known_tokens = defaultdict(float)
		for token in sentence:
			_token = token
			if token in self.redirs:
				token = self.redirs[token]
			if token in self.links:
				known_tokens[token] = max(
					known_tokens[token],
					self.weights[token],
					self.weights[_token]
				)
		if len(known_tokens) <= 1:
			return None, None
		source, target = sorted(
			sorted(
				enumerate(known_tokens.items()), key=lambda iws: iws[1][1], reverse=True
			)[0:2],  # selects the 2 words with the most significance (according to the weighting of x)
			key=lambda iws: iws[0]  # sort them by order of appearance in the sentence
		)
		return source[1][0], target[1][0]

	def score(self, path):
		# a sigmoid function
		return 1./(1.+np.exp(len(path)+self.alpha))

	def compute_weights(self, inputs: np.ndarray, labels: np.ndarray):
		print("Computing weights...", end=' ', flush=True)
		wset = set()
		for x in inputs:
			wset.update(x)
		revoc = {w: i for i, w in enumerate(wset)}
		words = np.full(shape=(len(wset), 2), fill_value=0.1)
		for x, label in zip(inputs, labels):
			for w in x:
				words[revoc[w], label] += 1
		# compute the entropy of each words
		weighted = np.sum(-np.log(words), axis=1)
		# store it in a dictionnary
		self.weights = defaultdict(float)
		self.weights.update({w: weighted[i] for i, w in enumerate(wset)})
		print("Done")

	def train(self, inputs: np.ndarray, labels: np.ndarray, _=10) -> float:
		self.compute_weights(inputs, labels)
		return Classifier.train(self, inputs, labels, k=1)

	def predict(self, inputs: np.ndarray) -> np.ndarray:
		res = np.full(shape=(inputs.shape[0],), fill_value=None)
		# for every input
		for i, x in enumerate(inputs):
			# we try to split it into individually scored sentences
			scores = []
			for sentence in get_sentences(x):
				source, target = self.get_source_target(sentence)
				path = self.shortest_path(source, target) if source else None
				if path:
					scores.append((self.score(path), min(self.weights[source], self.weights[target])))
			# unfortunately if no path is found we cannot score the input
			if scores:
				res[i] = 1 if sum((s*w for s, w in scores), 0.)/sum((w for _, w in scores), 0.) > .5 else 0
		return res

	def analyze(self, x, tokens, vocab) -> Tuple[float, str]:
		source, target = self.get_source_target(x[0])
		path = self.shortest_path(source, target) if source else None
		if path:
			return self.score(path), (
				f"<p>The path between <b>{source}</b> and <b>{target}</b> is:</p>" +
				f"<p>{' -> '.join(reversed([target]+path))}</p>"
			)
		else:
			return None, ""

	def save(self):
		with open("Logic/Classifiers/WikiInfoLinks/weights.pickle", "wb") as fweights:
			pickle.dump(self.weights, fweights, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=False)

	def load(self):
		with open("Logic/Classifiers/WikiInfoLinks/weights.pickle", "rb") as fweights:
			self.weights = pickle.load(fweights, fix_imports=False)
