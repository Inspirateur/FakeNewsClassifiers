from typing import Tuple
import pandas as pd
import numpy as np
folder = "Logic/Datasets"


class Data:
	def __init__(self, inputs: np.ndarray = np.ndarray(0), labels: np.ndarray = np.ndarray(0)):
		self.X: np.ndarray = inputs
		self.y: np.ndarray = labels
		self._classes = None
		self._counts = None

	def shuffle(self):
		p = np.random.permutation(len(self))
		self.X = self.X[p]
		self.y = self.y[p]

	def split(self, *fracs: float) -> Tuple["Data"]:
		assert sum(fracs, 0.) < 1
		self.shuffle()
		datas = []
		start = 0
		end = 0
		for frac in fracs:
			end += int(frac*len(self))
			datas.append(Data(self.X[start:end], self.y[start:end]))
			start = end
		datas.append(Data(self.X[end:], self.y[end:]))
		return tuple(datas)

	def samples(self, n=10) -> Tuple[np.ndarray, np.ndarray]:
		samples = np.random.choice(len(self), n)
		return self.X[samples], self.y[samples]

	@property
	def classes(self) -> Tuple[np.ndarray, np.ndarray]:
		if self._classes is None:
			self._classes, self._counts = np.unique(self.y, return_counts=True)
		return self._classes, self._counts

	def __add__(self, other: "Data"):
		if not len(self) & len(other):
			return Data(self.X, self.y)
		res = Data(
			np.concatenate((self.X, other.X)),
			np.concatenate((self.y, other.y))
		)
		if res._classes and other._classes:
			res._classes = np.unique(np.concatenate((self._classes, other._classes)))
			res._counts = self._counts+other._counts
		return res

	def __iadd__(self, other: "Data"):
		if len(self) & len(other):
			self.X = np.concatenate((self.X, other.X))
			self.y = np.concatenate((self.y, other.y))
			if other._classes:
				self._classes = np.unique(np.concatenate((self._classes, other._classes)))
				self._counts += other._counts

	def __getitem__(self, item):
		return Data(self.X[item], self.y[item])

	def __len__(self):
		return len(self.X)

	def __str__(self):
		classes, counts = self.classes
		counts = counts/counts.sum()
		ctxt = []
		for c, n in zip(classes, counts):
			ctxt.append(f"{c} {n:.1%}")
		X = np.vectorize(len)(self.samples(100)[0])
		avglen = int(np.mean(X))
		medlen = int(np.median(X))
		return f"{len(self)} examples | len {medlen}med {avglen}avg | "+", ".join(ctxt)


class Dataset:
	def __init__(self, train: Data, test: Data, valid: Data = Data()):
		self.train: Data = train
		self.test: Data = test
		self.valid: Data = valid
		self.classes, _ = self.train.classes
		cmap = {c: i for i, c in enumerate(self.classes)}
		cf = np.vectorize(cmap.get)
		self.train.y = cf(self.train.y)
		self.test.y = cf(self.test.y)
		if len(self.valid):
			self.valid.y = cf(self.valid.y)

	def detail(self):
		maxlen = len(str(max(len(self.train), len(self.valid), len(self.train))))
		return (
			f"train {' '*(maxlen-len(str(len(self.train))))}{self.train}\n"
			f"valid {' '*(maxlen-len(str(len(self.valid))))}{self.valid}\n"
			f"test  {' '*(maxlen-len(str(len(self.test))))}{self.test}\n"
		)

	def shuffle(self):
		trn_len = len(self.train)
		val_len = len(self.valid)
		total = self.train+self.valid+self.test
		total.shuffle()
		self.train = total[:trn_len]
		self.valid = total[trn_len:trn_len+val_len]
		self.test = total[trn_len+val_len:]


def get(dataset: str) -> Dataset:
	dataset = dataset.lower()
	if dataset == "liar":
		datas = []
		label_map = {
			"false": 0, "pants-fire": 0, "barely-true": 0,
			"half-true": 1, "mostly-true": 1, "true": 1
		}
		for name in ["train", "test", "valid"]:
			df = pd.read_csv(
				f"{folder}/LIAR/{name}.tsv", sep="\t", header=None, names=["label", "text"], usecols=[1, 2]
			)
			datas.append(Data(df["text"].to_numpy(), np.vectorize(label_map.get)(df["label"].to_numpy())))
		return Dataset(*datas)
	elif dataset == "fake-news-kaggle":
		df = pd.read_csv(f"{folder}/fake-news-kaggle/train.csv")
		df.dropna(inplace=True, axis=0)
		data = Data(df["text"].to_numpy(), np.vectorize(lambda l: int(l))(df["label"].to_numpy()))
		return Dataset(*data.split(.8))
	elif dataset == "reddit":
		raw = np.load(f"{folder}/Reddit/data_train.pkl", allow_pickle=True)
		data = Data(np.array(raw[0]), np.array(raw[1]))
		return Dataset(*data.split(.8, .1))
