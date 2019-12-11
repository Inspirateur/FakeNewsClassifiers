from typing import Tuple
from tensorflow.keras.layers import InputLayer, LSTM, Dense
from tensorflow.keras.models import Sequential, load_model
import numpy as np
from Logic.Classifiers.classifier import Classifier


class CLSTMClassifier(Classifier):
	model: Sequential

	def train(self, inputs: np.ndarray, labels: np.ndarray, _=10) -> float:
		self.model = Sequential()
		self.model.add(InputLayer(input_shape=inputs.shape[1:]))
		self.model.add(LSTM(units=1024, dropout=.2))
		self.model.add(Dense(1))
		self.model.compile(optimizer="adam", loss="mean_squared_error", metrics=['accuracy'])
		history = self.model.fit(inputs, labels, batch_size=16, epochs=2).history
		return history["accuracy"][-1]

	def predict(self, inputs: np.ndarray) -> np.ndarray:
		return self.model.predict(inputs)

	def analyze(self, x, tokens, vocab) -> Tuple[float, str]:
		pred = 1-self.model.predict(x)[0, 0]
		return pred, "<p>This is a black box for now.</p>"

	def save(self):
		self.model.save("Logic/Classifiers/CLSTM/model.h5")

	def load(self):
		self.model = load_model("Logic/Classifiers/CLSTM/model.h5")
