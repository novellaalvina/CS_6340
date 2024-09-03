# models.py

from sentiment_data import List
from sentiment_data import *
from utils import *
# import string
# import spacy
import random
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from collections import Counter

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """

        raise Exception("Don't call me, call my subclasses")

class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.vocab = indexer
        self.feature_vector = ""
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False):

        # pre processing of the words: lowercase, remove punctuation and stopwords
        sentence = [word.lower() for word in sentence]

        # building vocab unigram
        sentence_vocab = set(sentence)
        
        # building vocab and indexing
        if (add_to_indexer):                                            # training
            for word in sentence_vocab:
                index = self.vocab.add_and_get_index(word)
            return None
        else:                                                           # testing
            for word in sentence_vocab:
                if word not in self.vocab.objs_to_ints: 
                    sentence.remove(word)

        # building word counter from the sentence based on the vocab
        feature_counter = Counter(sentence)

        # building feature vector
        feature_vector = np.zeros(len(self.vocab), dtype=int)
        for feature in feature_counter.keys():
            index = self.vocab.index_of(feature)
            feature_vector[index] = feature_counter[feature]

        return feature_vector

class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.vocab = indexer
        self.feature_vector = ""
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False):

        # pre processing of the words: lowercase and remove punctuation 
        sentence = [word.lower() for word in sentence]

        # building vocab bigram
        sentence_bigram = []
        for i in range(len(sentence)-1):
            current = sentence[i] + " " + sentence[i+1]
            sentence_bigram.append(current)

        # building vocab index
        if (add_to_indexer):                                            # training
            for word in sentence_bigram:
                index = self.vocab.add_and_get_index(word)
            return None
        else:                                                           # testing
            for word in sentence_bigram:
                if word not in self.vocab.objs_to_ints: 
                    sentence_bigram.remove(word)

        # building word counter from the sentence based on the vocab
        feature_counter = Counter(sentence_bigram)

        # building feature vector
        self.feature_vector = np.zeros(len(self.vocab.objs_to_ints), dtype=int)
        for bigram in feature_counter.keys():
            index = self.vocab.index_of(bigram)
            self.feature_vector[index] = feature_counter[bigram]
        
        return self.feature_vector

class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.vocab = indexer
        self.feature_vector = ""
        self.document_frequency = {}
        self.docs_num = 0

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False):
        
        # # pre processing of the words: lowercase, remove punctuation and stopwords
        sentence = [word.lower() for word in sentence]
        # sentence = [word.translate(str.maketrans('', '', string.punctuation)) for word in sentence]
        stop_words = set(stopwords.words('english'))
        sentence = [word for word in sentence if word not in stop_words]

        # building term frequencies 
        # number of times a token(i.e word) occured in document(i.e sentence)
        term_freq = {}
        for word in sentence:
            if word in term_freq:
                term_freq[word] += 1
            else:
                term_freq[word] = 1
        
        # building document frequencies
        # number of times a token(i.e word) occured in the whole dataset(i.e N documents)
        sentence_vocab = set(sentence)
        for word in sentence_vocab:
            if word in self.document_frequency.keys():
                self.document_frequency[word] += 1
            else:
                self.document_frequency[word] = 1
                if (add_to_indexer):
                    self.vocab.add_and_get_index(word)
                else:
                    if word not in self.vocab.objs_to_ints:
                        sentence.remove(word)

        # building inverse document frequencies 
        # number of documents N / document frequency
        idf_vector = {}
        self.docs_num += 1  # total number of documents processed so far
        for word in sentence:
            idf = np.log(self.docs_num / self.document_frequency[word])
            idf_vector[word] = idf

        # building tf-idf 
        feature_vector = np.zeros(len(self.vocab), dtype=int)
        for word in sentence:
            if word in self.vocab.objs_to_ints:
                index = self.vocab.index_of(word)
                tf_idf = term_freq[word] * idf_vector[word]
                feature_vector[index] = tf_idf

        return feature_vector
    
class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")

class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weight, feat_extractor: FeatureExtractor):
        self.feature_extractor = feat_extractor
        self.weight = weight
        # raise Exception("Must be implemented")

    def predict(self, sentence: List[str]) -> int:
        feature_vector = self.feature_extractor.extract_features(sentence, False)

        # classification function (sigmoid_classification_func)
        probability_y_given_x = 1 / (1 + np.exp(-np.dot(feature_vector, self.weight)))
        if probability_y_given_x > 0.5: return 1
        else: return 0

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """

    epoch = 10
    learning_rate_alpha = 0.1

    # get vocab and its shape
    vocab = [feat_extractor.extract_features(ex.words, True) for ex in train_exs]
    num_features = len(feat_extractor.vocab)
    weights = np.zeros(num_features)

    for t in range(epoch):
       random.shuffle(train_exs)
       for ex in train_exs:
           sentence = set(ex.words)
           sentence_label = ex.label
           feature_vector = feat_extractor.extract_features(sentence, False) # add_to_indexer = False because we have build the vocab in line 178

           # classification function (sigmoid_classification_func)
           probability_y_given_x = 1 / (1+ np.exp(-np.dot(feature_vector, weights))) # y = 1

           # loss_func = -np.log(probability_y_given_x)
           # objective function to optimize model parameter

            # based on true label
           if sentence_label == 1:
                # min {neg log likelihood of the probability(y|w,x)} = min {loss func} -> gradient descent
                # compute gradient derivative of w of the loss func L(x_i, y_i, w) with respect to w
                dw = feature_vector * (probability_y_given_x-1)                                          # if y = 1: dw = f(x) (P(y=1|x) - 1)

           else:
                # min {neg log likelihood of the probability(y|w,x)} = min {loss func} -> gradient descent
                # compute gradient derivative of w of the loss func L(x_i, y_i, w) with respect to w
                dw = feature_vector * (probability_y_given_x)                                         # if y = 0: dw = f(x) (1 - P(y=0|x))

        #    dw = feature_vector * (1-sigmoid_classification_func - sentence_label)   
           
           # update the parameters
           weights = weights - learning_rate_alpha * dw  

    return LogisticRegressionClassifier(weights, feat_extractor=feat_extractor)

def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL or LR to run the appropriate system")
    return model