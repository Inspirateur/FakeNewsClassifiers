import numpy as np
from Logic.Classifiers.classifier import Classifier
import Logic.Datasets.datasets as datasets
from Logic.preprocessing import Vectorizer


class Model:
	def __init__(self, classifier: Classifier, dataset: str, vectorizer: Vectorizer = Vectorizer()):
		self.classifier = classifier
		self.dataset = dataset
		self.vectorizer = vectorizer

	def load_or_train(self):
		try:
			print(f"trying to load {self.classifier.__class__.__name__} from save...", end=" ", flush=True)
			self.classifier.load()
			print("Done")
		except (FileNotFoundError, OSError, ValueError):
			print("no save to load from")
			self.train()

	def train(self):
		print(f"training {self.classifier.__class__.__name__}...", end=" ", flush=True)
		inputs, labels = datasets.get(self.dataset)
		acc = self.classifier.train(self.vectorizer.fit_transform(inputs), labels)
		print(f"Acc = {acc:.1%}")
		print("Saving the model...", end=" ", flush=True)
		self.classifier.save()
		print("Done")

	def analyze(self, query):
		return self.classifier.analyze(
			self.vectorizer.transform(np.array([query], dtype=np.object)),
			self.vectorizer.tokenize(query)
		)
