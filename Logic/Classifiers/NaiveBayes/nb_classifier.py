from typing import Tuple
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from Logic.Classifiers.classifier import Classifier


def arg_top_n(array: np.ndarray, n: int):
	if n >= len(array):
		return np.arange(len(array))
	return array.argpartition(-n)[n:]


def html_highlight(probs: np.ndarray, tokens):
	topmask = np.zeros(len(probs), dtype=np.bool)
	topmask[arg_top_n(probs, 20)] = True
	threshold = np.amax(probs[topmask])/3.
	probmask = topmask & (probs > threshold)
	words = []
	for i, token in enumerate(tokens):
		if probmask[i]:
			words.append(token)
		elif words and words[-1] != "...":
			words.append("...")
	return (
		f"<p>The most important words were:</p>"
		f"<p>«{' '.join(words)}»</p>"
	)


class NBClassifier(Classifier):
	model: MultinomialNB

	def __init__(self, alpha=.2):
		self.alpha = alpha

	def _train(self, inputs: np.ndarray, labels: np.ndarray):
		self.model = MultinomialNB(alpha=self.alpha)
		self.model.fit(inputs, labels)

	def predict(self, inputs: np.ndarray) -> np.ndarray:
		return self.model.predict(inputs)

	def analyze(self, x, tokens, vocab) -> Tuple[float, str]:
		probs = np.zeros(shape=len(tokens))
		for i, token in enumerate(tokens):
			if token in vocab:
				probs[i] = x[0, vocab[token]]
		return self.model.predict_proba(x)[0, 0], html_highlight(probs, tokens)
