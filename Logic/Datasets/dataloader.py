from Logic.Datasets.dataset import Data, Dataset
import pandas as pd
import numpy as np
folder = "Logic/Datasets"


def get(dataset: str) -> Dataset:
	dataset = dataset.lower()
	if dataset == "liar":
		datas = []
		# we reduce the labels to only 3, because they are not well separated
		label_map = {
			"false": "false", "pants-fire": "false", "barely-true": "half true",
			"half-true": "half true", "mostly-true": "half true", "true": "true"
		}
		for name in ["train", "valid", "test"]:
			df = pd.read_csv(
				f"{folder}/LIAR/{name}.tsv", sep="\t", header=None, names=["label", "text"], usecols=[1, 2]
			)
			datas.append(Data(df["text"].to_numpy(), np.vectorize(label_map.get)(df["label"].to_numpy())))
		return Dataset(*datas, name="liar", classes={"false": 0, "half true": 0.5, "true": 1})
	elif dataset == "fake-news-kaggle":
		df = pd.read_csv(f"{folder}/fake-news-kaggle/train.csv")
		df["text"] = df["text"].str.strip()
		df["text"].replace('', np.nan, inplace=True)
		df.dropna(inplace=True, axis=0)
		data = Data(df["text"].to_numpy(), df["label"].to_numpy())
		# this dataset uses 0 for reliable and 1 for unreliable
		return Dataset(*data.split(.8, .1), name="fake_news_kaggle", classes=[True, False])
	elif dataset == "fake-news-corpus":
		df = pd.read_csv(
			f"{folder}/Fake-News-corpus/news_shuffled.csv", header=0,
			names=["label", "text"], usecols=[3, 9], dtype=str,
			nrows=100_000
		)
		df["text"] = df["text"].str.strip()
		df["text"].replace('', np.nan, inplace=True)
		df["label"] = df["label"].str.strip()
		df["label"].replace('', np.nan, inplace=True)
		df["label"].replace('unknown', np.nan, inplace=True)
		df.dropna(inplace=True, axis=0)
		data = Data(df["text"].to_numpy(), df["label"].to_numpy())
		# this dataset has a very complete sets of label
		return Dataset(*data.split(.9, .05), name="fake_news_corpus", classes={
			"fake": 0, "satire": 0, "bias": 0.2, "conspiracy": 0, "state": 0, "junksci": 0,
			"hate": 0, "clickbait": .5, "unreliable": .5, "rumor": .5, "political": .7, "reliable": 1
		})
	elif dataset == "reddit":
		raw = np.load(f"{folder}/Reddit/data_train.pkl", allow_pickle=True)
		data = Data(np.array(raw[0]), np.array(raw[1]))
		# this has very little to do with fake news,
		# it's just a text classification benchmark
		return Dataset(*data.split(.8, .1), name="reddit")
