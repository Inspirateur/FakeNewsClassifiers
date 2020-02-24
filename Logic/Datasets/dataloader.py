from Logic.Datasets.dataset import Data, Dataset
import pandas as pd
import numpy as np
folder = "Logic/Datasets"


def get(dataset: str) -> Dataset:
	dataset = dataset.lower()
	if dataset == "liar":
		datas = []
		label_map = {
			"false": "False", "pants-fire": "False", "barely-true": "False",
			"half-true": "True", "mostly-true": "True", "true": "True"
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
		data = Data(df["text"].to_numpy(), df["label"].to_numpy())
		return Dataset(*data.split(.8, .1), classes=["True", "False"])
	elif dataset == "reddit":
		raw = np.load(f"{folder}/Reddit/data_train.pkl", allow_pickle=True)
		data = Data(np.array(raw[0]), np.array(raw[1]))
		return Dataset(*data.split(.8, .1))
