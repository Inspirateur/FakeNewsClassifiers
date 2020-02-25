from typing import Tuple
from tensorflow.keras.layers import InputLayer, LSTM, Dense
from tensorflow.keras.models import Sequential, load_model
import numpy as np
from Logic.Classifiers.classifier import Classifier
from Logic.preprocessing import GloVeVectorizer, tokenize


class CLSTMClassifier(Classifier):
	model: Sequential
	vec: GloVeVectorizer

	def train(self) -> float:
		self.vec = GloVeVectorizer(tokenize)
		self.model = Sequential()
		X = self.vec.fit_transform(self.d.train.X)
		self.model.add(InputLayer(input_shape=X.shape[1:]))
		self.model.add(LSTM(units=350, dropout=.2))
		self.model.add(Dense(1))
		self.model.compile(optimizer="adam", loss="mean_squared_error", metrics=['accuracy'])
		self.model.fit(X, self.d.train.y, batch_size=16, epochs=15, validation_split=.05)

	def predict(self, inputs: np.ndarray) -> np.ndarray:
		return self.model.predict(inputs)

	def analyze(self, x) -> Tuple[float, str]:
		pred = 1-self.model.predict(x)[0, 0]
		return pred, "<p>This is a black box for now.</p>"

	def save(self):
		self.model.save("Logic/Classifiers/CLSTM/model.h5")

	def load(self):
		self.model = load_model("Logic/Classifiers/CLSTM/model.h5")
