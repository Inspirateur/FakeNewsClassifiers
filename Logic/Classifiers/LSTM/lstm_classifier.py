import json
from typing import Tuple
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
from tensorflow.keras.models import Sequential, load_model
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant
import numpy as np
from Logic.Classifiers.classifier import Classifier


class LSTMClassifier(Classifier):
	model: Sequential
	vec: Tokenizer
	maxlen: int

	def train(self) -> float:
		d = 100
		# get glove embeds
		glove_embeds = {}
		with open(f"Logic/ExternalData/glove.{d}d.txt") as f:
			for line in f:
				values = line.split()
				word = values[0]
				coefs = np.asarray(values[1:], dtype='float32')
				glove_embeds[word] = coefs
		# make vocabulary
		self.maxlen = 60
		vocablen = 40_000
		self.vec = Tokenizer(num_words=vocablen, oov_token='<unw>')
		self.vec.fit_on_texts(self.d.train.X)
		trn_X = pad_sequences(self.vec.texts_to_sequences(self.d.train.X), self.maxlen)
		vocablen = min(len(self.vec.word_index), vocablen)+1
		# make embedding matrix
		embedding_matrix = np.zeros((vocablen, d))
		for token, i in self.vec.word_index.items():
			if i >= vocablen:
				break
			if token in glove_embeds:
				embedding_matrix[i] = glove_embeds[token]
			else:
				embedding_matrix[self.vec.word_index[token]] = np.random.randn(d)
		# make the model
		self.model = Sequential()
		self.model.add(Embedding(
			vocablen, d, input_length=self.maxlen,
			embeddings_initializer=Constant(embedding_matrix),
			trainable=True
		))
		self.model.add(Bidirectional(LSTM(64, dropout=.2, return_sequences=True)))
		self.model.add(Bidirectional(LSTM(32, dropout=.2)))
		self.model.add(Dense(len(self.d.classes), activation="softmax"))
		self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['sparse_categorical_accuracy'])
		self.model.summary()
		# train the model
		val_X = pad_sequences(self.vec.texts_to_sequences(self.d.valid.X), self.maxlen)
		self.model.fit(trn_X, self.d.train.y, batch_size=16, epochs=3, validation_data=(val_X, self.d.valid.y))

	def predict(self, inputs: np.ndarray) -> np.ndarray:
		return np.argmax(
			self.model.predict(
				pad_sequences(self.vec.texts_to_sequences(inputs), maxlen=self.maxlen)
			),
			axis=1
		)

	def analyze(self, query) -> Tuple[float, str]:
		x = pad_sequences(self.vec.texts_to_sequences(np.array([query])), self.maxlen)
		pred = self.model.predict(x)[0]
		score = self.d.score(pred)
		return score, f"<p>Labelled {self.d.classes[list(self.d.classes.keys())[np.argmax(pred)]]}</p>"

	def save(self):
		self.model.save(f"Logic/Classifiers/LSTM/save_{self.d.name}.h5")
		with open(f"Logic/Classifiers/LSTM/vocab_{self.d.name}.json", "w") as ftoken:
			ftoken.write(self.vec.to_json())
		with open(f"Logic/Classifiers/LSTM/maxlen_{self.d.name}.txt", "w") as fmaxlen:
			fmaxlen.write(str(self.maxlen))

	def load(self):
		with open(f"Logic/Classifiers/LSTM/vocab_{self.d.name}.json", "r") as ftoken:
			self.vec = tokenizer_from_json(ftoken.read())
		with open(f"Logic/Classifiers/LSTM/maxlen_{self.d.name}.txt", "r") as fmaxlen:
			self.maxlen = int(fmaxlen.read().strip())
		self.model = load_model(f"Logic/Classifiers/LSTM/save_{self.d.name}.h5")
