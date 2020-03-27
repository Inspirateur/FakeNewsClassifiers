import numpy as np
from sklearn.naive_bayes import MultinomialNB
from Logic.Classifiers.classifier import Classifier
from Logic.preprocessing import TFIDFVectorizer, tokenize


class NBClassifier(Classifier):
	model: MultinomialNB
	vec: TFIDFVectorizer

	def __init__(self, data: str, alpha=.2):
		Classifier.__init__(self, data)
		self.alpha = alpha

	def train(self):
		self.vec = TFIDFVectorizer(tokenize)
		self.model = MultinomialNB(alpha=self.alpha)
		self.model.fit(self.vec.fit_transform(self.d.train.X), self.d.train.y)

	def predict(self, inputs: np.ndarray) -> np.ndarray:
		return self.model.predict(self.vec.transform(inputs))
