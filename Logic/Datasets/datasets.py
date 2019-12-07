from typing import Tuple
import pandas as pd
import numpy as np


def get(dataset: str) -> Tuple[np.ndarray, np.ndarray]:
	if dataset == "fake-news":
		data = pd.read_csv(
			"Logic/Datasets/LIAR/train.tsv", sep="\t", header=None, names=["label", "text"], usecols=[1, 2]
		)
		return data["text"].to_numpy(), np.vectorize(lambda l: 0 if l == "false" else 1)(data["label"].to_numpy())
	elif dataset == "fake-news-kaggle":
		data = pd.read_csv("Logic/Datasets/fake-news-kaggle/train.csv")
		return data["text"].apply(lambda x: np.str_(x)), np.vectorize(lambda l: int(l))(data["label"].to_numpy())
