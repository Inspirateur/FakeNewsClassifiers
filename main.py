from Logic.preprocessing import tokenize
datas = ["reddit", "fake-news-kaggle", "LIAR"]


def wikimodel(data: str):
	from Logic.Classifiers.WikiInfoLinks.wikilinks_classifier import WikiLinksClassifier
	from Logic.preprocessing import Vectorizer

	return WikiLinksClassifier(data, Vectorizer(tokenize))


def nbmodel(data: str):
	from Logic.Classifiers.NaiveBayes.nb_classifier import NBClassifier
	from Logic.preprocessing import TFIDFVectorizer

	return NBClassifier(data, TFIDFVectorizer(tokenize))


def deeplmodel(data: str):
	from Logic.Classifiers.CLSTM.clstm_classifier import CLSTMClassifier
	from Logic.preprocessing import GloVeVectorizer

	return CLSTMClassifier(data, GloVeVectorizer(tokenize))


def transfomodel(data: str):
	from Logic.Classifiers.Transformer.transformer_classifier import TransformerClassifier
	from Logic.preprocessing import Vectorizer

	return TransformerClassifier(data, Vectorizer())


m = transfomodel("reddit")
m.evaluate(1)
