from typing import Tuple
import pandas as pd
import numpy as np

data = pd.concat([pd.read_csv("Logic/Datasets/fake-news-kaggle/train.csv"), pd.read_csv("Logic/Datasets/fake-news-kaggle/test.csv")], axis=0)
data["text"].apply(lambda x: np.str_(x))
data.dropna(inplace=True, axis=0)
np.vectorize(lambda l: int(l))(data["label"].to_numpy())




print("a")

