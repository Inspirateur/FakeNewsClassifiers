from Logic.Classifiers.CLSTM.clstm_classifier import CLSTMClassifier
from Logic.model import Model
from Logic.preprocessing import tokenize, GloVeVectorizer


m = Model(CLSTMClassifier(), "fake-news-kaggle", GloVeVectorizer(tokenize))
m.load_or_train()
m.analyze("barack obama is a muslim.")
