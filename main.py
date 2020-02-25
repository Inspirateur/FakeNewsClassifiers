from Logic.Classifiers.WikiInfoLinks.wikilinks_classifier import WikiLinksClassifier
from Logic.Classifiers.NaiveBayes.nb_classifier import NBClassifier
from Logic.Classifiers.CLSTM.clstm_classifier import CLSTMClassifier
from Logic.Classifiers.Transformer.transformer_classifier import TransformerClassifier


# datasets: reddit, fake-news-kaggle, LIAR
m = TransformerClassifier("fake-news-kaggle")
m.train()
m.evaluate()
m.save()
# m.load()
# m.evaluate()
