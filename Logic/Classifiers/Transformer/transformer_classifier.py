from typing import Tuple
import numpy as np
from Logic.Classifiers.classifier import Classifier
# from transformers.modeling_distilbert import DistilBertForSequenceClassification
# import torch
import ktrain
from ktrain import text
from ktrain.text.predictor import TextPredictor
# TODO: use huggingface instead


class TransformerClassifier(Classifier):
	predictor: TextPredictor
	t: text

	def train(self) -> float:
		self.t = text.Transformer('distilbert-base-uncased', classes=list(range(len(self.d.classes))))
		trn = self.t.preprocess_train(self.d.train.X, self.d.train.y)
		val = self.t.preprocess_test(self.d.valid.X, self.d.valid.y)
		model = self.t.get_classifier()
		learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=16)
		learner.fit_onecycle(lr=4e-5, epochs=3)
		self.predictor = ktrain.get_predictor(learner.model, self.t)

	def predict(self, inputs: np.ndarray) -> np.ndarray:
		return self.predictor.predict(inputs)

	def analyze(self, query) -> Tuple[float, str]:
		probas = self.predictor.predict(query, return_proba=True)
		# label = list(self.d.classes.keys())[np.argmax([probas])]
		explain = self.predictor.explain(query)
		return self.d.score(probas), explain

	def save(self):
		self.predictor.save(f"Logic/Classifiers/Transformer/Saves/save_{self.d.name}")

	def load(self):
		self.predictor = ktrain.load_predictor(f"Logic/Classifiers/Transformer/Saves/save_{self.d.name}")
