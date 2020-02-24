from typing import Tuple
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from Logic.Classifiers.classifier import Classifier
from Logic.preprocessing import Vectorizer


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

	def __init__(self, data: str, vectorizer: Vectorizer = None, alpha=.2):
		Classifier.__init__(self, data, vectorizer)
		self.alpha = alpha

	def train(self):
		self.model = MultinomialNB(alpha=self.alpha)
		self.model.fit(self.vec.fit_transform(self.d.train.X), self.d.train.y)

	def predict(self, inputs: np.ndarray) -> np.ndarray:
		return self.model.predict(self.vec.transform(inputs))

	def analyze(self, query: str) -> Tuple[float, str]:
		tokens = self.vec.tokenize(query)
		x = self.vec.transform(tokens)
		probs = np.zeros(shape=len(tokens))
		for i, token in enumerate(tokens):
			if token in self.vec.vocab:
				probs[i] = x[0, self.vec.vocab[token]]
		return self.model.predict_proba(x)[0, 0], html_highlight(probs, tokens)
