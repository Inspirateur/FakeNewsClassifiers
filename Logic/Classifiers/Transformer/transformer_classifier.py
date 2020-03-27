import os
from typing import Tuple
import sys
import numpy as np
from Logic.Classifiers.classifier import Classifier
from transformers.modeling_distilbert import DistilBertForSequenceClassification
from transformers import BertTokenizer
import torch
from tqdm import tqdm, trange
_path = "Logic/Classifiers/Transformer/"


def var_batch(X, y=None, batch_size=8):
	# batches of variable size
	i = 0
	while i+batch_size < len(X):
		batch = X[i:i+batch_size]
		maxlen = max(len(sentence) for sentence in batch)
		if maxlen < 200:
			res = torch.zeros(size=(batch_size, maxlen), dtype=torch.int64)
			for j, sentence in enumerate(batch):
				for k, t in enumerate(sentence):
					res[j, k] = t
			if y is not None:
				yield res, torch.tensor(y[i:i+batch_size], dtype=torch.int64)
			else:
				yield res
		i += batch_size


class TransformerClassifier(Classifier):
	tokenizer: BertTokenizer
	model: DistilBertForSequenceClassification
	device: torch.device

	def __init__(self, data: str):
		Classifier.__init__(self, data)
		self.save_dir = _path+f"save_{self.d.name}/"
		if torch.cuda.is_available():
			print("Using the GPU")
			self.device = torch.device("cuda")
		else:
			print("Using the CPU")
			self.device = torch.device("cpu")

	def _tokenize(self, inputs):
		bos = self.tokenizer.special_tokens_map["cls_token"]  # `[CLS]`
		sep = self.tokenizer.special_tokens_map["sep_token"]  # `[SEP]`
		return [
			self.tokenizer.encode(
				f"{bos} {text.strip()} {sep}",
				add_special_tokens=False
			)
			for text in inputs
		]

	def train(self) -> float:
		"""
		tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
		model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased')
		input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)
		labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
		outputs = model(input_ids, labels=labels)
		loss, logits = outputs[:2]
		"""
		self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
		self.model = DistilBertForSequenceClassification.from_pretrained(
			"distilbert-base-uncased", num_labels=len(self.d.classes)
		).to(self.device)
		no_decay = ["bias", "LayerNorm.weight"]
		optimizer = torch.optim.Adam(
			[p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
			lr=2e-5
		)
		loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.d.weights, dtype=torch.float).to(self.device))
		trn_X = self._tokenize(self.d.train.X)
		batch_size = 8
		steps = len(trn_X)//batch_size
		avg_window = 50
		for epoch in range(1):
			print(f"Epoch {epoch+1}:")
			running_loss = 0.0
			epoch_iter = tqdm(
				var_batch(trn_X, self.d.train.y, batch_size=batch_size),
				desc="Iteration", total=steps, file=sys.stdout
			)
			self.model.train()
			self.model.zero_grad()
			for i, (X, y) in enumerate(epoch_iter):
				logits, = self.model(X.to(self.device))
				loss = loss_fn(logits, y.to(self.device))
				loss.backward()
				optimizer.step()
				self.model.zero_grad()
				# print statistics
				running_loss += loss.item()
				if i % avg_window == avg_window-1:  # print every 2000 mini-batches
					epoch_iter.set_postfix_str(f"loss: {running_loss/avg_window:.3f}")
					running_loss = 0.0

			self.model.eval()
			val_loss = 0.0
			val_count = 0
			for X, y in var_batch(self._tokenize(self.d.valid.X), self.d.valid.y, batch_size):
				logits, = self.model(X.to(self.device))
				val_loss += loss_fn(logits, y.to(self.device)).item()
				val_count += 1
			val_loss /= val_count
			print(f"val_loss: {val_loss:.3f}\n")

	def predict(self, inputs: np.ndarray) -> np.ndarray:
		self.model.eval()
		res = np.empty(shape=(len(inputs),), dtype=np.int)
		for i, X in enumerate(self._tokenize(inputs)):
			logits, = self.model(torch.tensor(X).unsqueeze(0).to(self.device))
			res[i] = torch.argmax(logits, dim=1)
		return res

	def analyze(self, query) -> Tuple[float, str]:
		self.model.eval()
		# prepare input
		X = torch.tensor(self._tokenize([query])).to(self.device)
		# get logit as numpy ndarray
		logits, = self.model(X)
		logits = logits.view(-1).cpu().detach().numpy()
		# apply softmax to convert to proba
		elogits = np.exp(logits)
		logits = elogits/elogits.sum()
		# compute the trust score (dataset dependant)
		score = self.d.score(logits)
		# return the score and the label
		return score, f"<p>Labelled <b>{list(self.d.classes.keys())[np.argmax(logits)]}</b></p>"

	def save(self):
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)
		self.model.save_pretrained(self.save_dir)
		self.tokenizer.save_pretrained(self.save_dir)

	def load(self):
		self.model = DistilBertForSequenceClassification.from_pretrained(self.save_dir)
		self.model.to(self.device)
		self.tokenizer = BertTokenizer.from_pretrained(self.save_dir)
