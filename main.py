from Logic.Classifiers.classifier import Classifier
from Logic.Classifiers.NaiveBayes.nb_classifier import NBClassifier
from Logic.Classifiers.LSTM.lstm_classifier import LSTMClassifier
from Logic.Classifiers.Transformer.transformer_classifier import TransformerClassifier
import csv
import sys


def sample(chunk_size=100, freq=5):
	import pandas as pd
	from pandas.io.parsers import TextFileReader
	csv.field_size_limit(sys.maxsize)
	chunks: TextFileReader = pd.read_csv(
		f"Logic/Datasets/Fake-News-corpus/news_cleaned_2018_02_13.csv",
		index_col=0, chunksize=chunk_size, nrows=8_400_000,
		engine="python"
	)
	i = 0
	sampled = []
	for chunk in chunks:
		if i % freq == 0:
			sampled.append(chunk)
			if i > 0 and i % (freq*100) == 0:
				print('.', end='', flush=True)
			if i > 0 and i % (freq*1000) == 0:
				print(f" {i*chunk_size:,} rows")
		i += 1
	print()
	print("Concatenating")
	df: pd.DataFrame = pd.concat(sampled)
	print("Shuffling")
	df = df.sample(frac=1).reset_index(drop=True)
	print("Writing to file")
	df.to_csv(f"Logic/Datasets/Fake-News-corpus/news_shuffled.csv")


def test_train(model: Classifier):
	model.train()
	model.evaluate()
	model.save()


def test_load(model: Classifier):
	model.load()
	model.evaluate()


# datasets: reddit, fake-news-kaggle, LIAR, fake-news-corpus
test_train(TransformerClassifier("fake-news-corpus"))
