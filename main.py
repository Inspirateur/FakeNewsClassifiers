from Logic.model import Model
from Logic.preprocessing import tokenize


def wikimodel():
	from Logic.Classifiers.WikiInfoLinks.wikilinks_classifier import WikiLinksClassifier
	from Logic.preprocessing import TFIDFVectorizer

	m = Model(WikiLinksClassifier(), "fake-news-kaggle", TFIDFVectorizer(tokenize))
	m.load_or_train()
	return m


def nbmodel():
	from Logic.Classifiers.NaiveBayes.nb_classifier import NBClassifier
	from Logic.preprocessing import TFIDFVectorizer

	m = Model(NBClassifier(), "fake-news-kaggle", TFIDFVectorizer(tokenize))
	m.load_or_train()
	return m


def deeplmodel():
	from Logic.Classifiers.CLSTM.clstm_classifier import CLSTMClassifier
	from Logic.preprocessing import GloVeVectorizer

	m = Model(CLSTMClassifier(), "fake-news-kaggle", GloVeVectorizer(tokenize))
	m.load_or_train()
	return m


a = wikimodel().analyze("barack obama is a muslim")
