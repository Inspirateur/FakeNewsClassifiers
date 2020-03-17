from collections import defaultdict
import re
from typing import Callable
import csv
import pandas as pd
from keras_preprocessing.text import Tokenizer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
regtoken = re.compile(r"(?u)\b\w+\b")


def tokenize(sentence: str):
	return regtoken.findall(sentence.lower())


class Vectorizer:
	def __init__(self, tokenizer: Callable = lambda x: x):
		self.tokenize = tokenizer
		self.vocab = defaultdict(int)

	def fit_transform(self, inputs: np.ndarray) -> np.ndarray:
		return self.transform(inputs)

	def transform(self, inputs: np.ndarray) -> np.ndarray:
		res = np.ndarray(shape=(inputs.shape[0],), dtype=np.object)
		for i in range(inputs.shape[0]):
			res[i] = self.tokenize(inputs[i])
		return res


class TFIDFVectorizer(Vectorizer):
	count_vect: CountVectorizer
	tfidf_transf: TfidfTransformer

	def fit_transform(self, inputs: np.ndarray) -> np.ndarray:
		self.count_vect = CountVectorizer(tokenizer=self.tokenize, lowercase=False)
		self.tfidf_transf = TfidfTransformer()
		counts = self.count_vect.fit_transform(inputs)
		self.vocab = self.count_vect.vocabulary_
		return self.tfidf_transf.fit_transform(counts)

	def transform(self, inputs: np.ndarray) -> np.ndarray:
		counts = self.count_vect.transform(inputs)
		return self.tfidf_transf.transform(counts)


class GloVeVectorizer(Vectorizer):
	vect: Tokenizer
	embedding: pd.DataFrame

	def __init__(self, tokenizer: Callable = lambda x: x, maxlen: int = 60):
		Vectorizer.__init__(self, tokenizer)
		self.maxlen = maxlen
		self.dims = 100
		self.vocab = defaultdict(lambda: np.zeros(shape=(self.dims,)))
		print("Loading GloVe embedding...", end=' ')
		df = pd.read_csv(
			f"Logic/ExternalData/glove.{self.dims}d.txt",
			sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE
		)
		self.vocab.update({w: vec.values for w, vec in df.T.items()})
		print("Done")

	def transform(self, inputs: np.ndarray) -> np.ndarray:
		res = np.zeros(shape=(inputs.shape[0], self.maxlen, self.dims))
		for i in range(inputs.shape[0]):
			for j, token in enumerate(self.tokenize(inputs[i])[:self.maxlen]):
				res[i, j] = self.vocab[token]
		return res
