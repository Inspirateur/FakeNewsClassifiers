from typing import Tuple
import numpy as np
from Logic.Classifiers.classifier import Classifier
from transformers.modeling_distilbert import DistilBertForSequenceClassification
import torch
import ktrain
from ktrain import text


class TransformerClassifier(Classifier):
	model: text.Transformer
	learner: ktrain.GenLearner

	def train(self) -> float:
		t = text.Transformer('distilbert-base-uncased', classes=list(range(len(self.d.classes))))
		X = self.vec.fit_transform(self.d.train.X)
		trn = t.preprocess_train(X, self.d.train.y)
		self.model = t.get_classifier()
		self.learner = ktrain.get_learner(self.model, train_data=trn, batch_size=8)
		self.learner.fit_onecycle(lr=4e-5, epochs=2)

	def predict(self, inputs: np.ndarray) -> np.ndarray:
		return self.learner.predict(self.vec.transform(inputs))

	def analyze(self, x) -> Tuple[float, str]:
		pass
