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
	last = -1
	words = []
	cache = {}
	i = 0
	for j, token in enumerate(tokens):
		if (token in cache and cache[token]) or probmask[i]:
			if j-1 > last:
				words.append("...")
			words.append(tokens[j])
			last = j
			cache[token] = True
		else:
			cache[token] = False
		i = len(cache)
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

	def analyze(self, x, tokens) -> Tuple[float, str]:
		preds = self.model.predict_proba(x)[0]
		pred_i = np.argmax(preds)
		probs = np.exp(self.model.feature_log_prob_[pred_i, x.indices])
		return preds[0], html_highlight(probs, tokens)

	def save(self):
		pass

	def load(self):
		# this model is quick to train so we don't bother save/load-ing it
		raise FileNotFoundError
