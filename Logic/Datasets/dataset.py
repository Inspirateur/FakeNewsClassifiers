from typing import List, Tuple
import numpy as np
sets = ["train", "valid", "test"]


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
		if not len(other):
			return Data(other.X, other.y)
		if not len(self):
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
		return self

	def __getitem__(self, item):
		return Data(self.X[item], self.y[item])

	def __len__(self):
		return len(self.X)


class Dataset:
	def __init__(self, train: Data, test: Data, valid: Data, classes: List[str] = None):
		self.train: Data = train
		self.test: Data = test
		self.valid: Data = valid
		if classes:
			self.classes = np.array(classes)
		else:
			self.classes = np.unique(self.train.y)
			cmap = {c: i for i, c in enumerate(self.classes)}
			cf = np.vectorize(cmap.get)
			self.train.y = cf(self.train.y)
			self.test.y = cf(self.test.y)
			self.valid.y = cf(self.valid.y)

	def __getitem__(self, item: str) -> Data:
		if item == "train":
			return self.train
		if item == "valid":
			return self.valid
		if item == "test":
			return self.test

	def shuffle(self):
		trn_len = len(self.train)
		val_len = len(self.valid)
		total = self.train+self.valid+self.test
		total.shuffle()
		self.train = total[:trn_len]
		self.valid = total[trn_len:trn_len+val_len]
		self.test = total[trn_len+val_len:]

	def detail(self):
		header = ["", "examples", "med_len", "avg_len"]+self.classes.tolist()
		table = np.empty(shape=(len(sets)+1, len(header)), dtype=np.object)
		for i, col in enumerate(header):
			table[0, i] = col
		for i, s in enumerate(sets):
			table[i+1, 0] = s
			table[i+1, 1] = str(len(self[s]))
			X = np.vectorize(len)(self[s].samples(100)[0])
			table[i+1, 2] = str(int(np.median(X)))
			table[i+1, 3] = str(int(np.mean(X)))
			classes, counts = self[s].classes
			counts = counts/counts.sum()
			for c, count in zip(classes, counts):
				table[i+1, c+4] = str(round(count, 2))
		lentable = np.vectorize(len)(table)
		lentable = -lentable+np.amax(lentable, axis=0)+2
		text = ""
		for i in range(table.shape[0]):
			for j in range(table.shape[1]):
				text += ' '*lentable[i, j]+table[i, j]
			text += '\n'
		return text
