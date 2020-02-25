from typing import Tuple
import numpy as np
from Logic.Classifiers.classifier import Classifier
from Logic.preprocessing import Vectorizer, tokenize
# from transformers.modeling_distilbert import DistilBertForSequenceClassification
# import torch
import ktrain
from ktrain import text
from ktrain.predictor import Predictor


class TransformerClassifier(Classifier):
	predictor: Predictor
	t: text
	vec: Vectorizer

	def train(self) -> float:
		self.vec = Vectorizer(tokenize)
		self.t = text.Transformer('distilbert-base-uncased', classes=list(range(len(self.d.classes))))
		X = self.vec.transform(self.d.train.X)
		X_val = self.vec.transform(self.d.valid.X)
		trn = self.t.preprocess_train(X, self.d.train.y)
		val = self.t.preprocess_test(X_val, self.d.valid.y)
		model = self.t.get_classifier()
		learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=16)
		learner.fit_onecycle(lr=4e-5, epochs=1)
		self.predictor = ktrain.get_predictor(learner.model, self.t)

	def predict(self, inputs: np.ndarray) -> np.ndarray:
		return self.predictor.predict(self.vec.transform(inputs))

	def analyze(self, x) -> Tuple[float, str]:
		pass

	def save(self):
		self.predictor.save("Logic/Classifiers/Transformer/save")

	def load(self):
		self.vec = Vectorizer(tokenize)
		self.predictor = ktrain.load_predictor("Logic/Classifiers/Transformer/save")
