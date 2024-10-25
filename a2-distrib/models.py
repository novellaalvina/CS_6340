# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FFNN(nn.Module):

    def __init__(self, input, hidden, output):
        super(FFNN, self).__init__()
        self.V = nn.Linear(input, hidden)
        self.g = nn.Tanh()
        # self.g = nn.ReLU()
        self.W = nn.Linear(hidden, output)
        self.log_softmax = nn.LogSoftmax(dim=0)
        self.softmax = nn.Softmax(dim=0)
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)
        # Initialize with zeros instead
        # nn.init.zeros_(self.V.weight)
        # nn.init.zeros_(self.W.weight)
        self.counter = 1

    def forward(self, x):
        return self.log_softmax((self.W(self.g(self.V(x)))))


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """

    def __init__(self, word_embeddings: WordEmbeddings, FFNN: FFNN):
        self.word_embeddings = word_embeddings
        self.nn = FFNN

    def predict(self, words: List[str], has_typos: bool):
        avg_tensor = form_input(avg_embeddings(words, self.word_embeddings))
        probabilities = self.nn.forward(avg_tensor)
        return torch.argmax(probabilities).item()


def form_input(x) -> torch.Tensor:
    """
    Form the input to the neural network. In general this may be a complex function that synthesizes multiple pieces
    of data, does some computation, handles batching, etc.

    :param x: a [num_samples x inp] numpy array containing input data
    :return: a [num_samples x inp] Tensor
    """
    return torch.from_numpy(x).float()


def avg_embeddings(words, word_embeddings: WordEmbeddings):
    """
    Calculate the vector average of the embeddings associated with words.
    """
    length = len(words)
    sum = np.zeros(word_embeddings.get_embedding_length())
    for word in words:
        embedding = word_embeddings.get_embedding(word)
        print(embedding)
        sum += embedding

    return sum / length


def train_deep_averaging_network(
    args,
    train_exs: List[SentimentExample],
    dev_exs: List[SentimentExample],
    word_embeddings: WordEmbeddings,
    train_model_for_typo_setting: bool,
) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """
    # Define some constants
    batch_size = args.batch_size
    # Inputs are of size embedding length
    feat_vec_size = word_embeddings.get_embedding_length()
    # Let's use 4 hidden units
    embedding_size = 4
    # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
    # slightly more compact code for the binary case is possible.
    num_classes = 2

    enable_batching = not train_model_for_typo_setting and batch_size > 1

    # RUN TRAINING AND TEST
    num_epochs = args.num_epochs
    ffnn = FFNN(feat_vec_size, embedding_size, num_classes)
    initial_learning_rate = 0.001
    optimizer = optim.Adam(ffnn.parameters(), lr=initial_learning_rate)

    # Batching
    if enable_batching:
        for _ in range(num_epochs):
            random.shuffle(train_exs)

            num_of_batch = np.ceil(len(train_exs) / batch_size)
            batches = np.array_split(train_exs, num_of_batch)

            for batch in batches:
                gold_label = np.zeros((batch_size, num_classes))
                batch_result = np.zeros((batch_size, feat_vec_size))

                for i in range(len(batch)):
                    example = batch[i]
                    
                    x_avg = avg_embeddings(example.words, word_embeddings)
                    x = form_input(x_avg)
                    batch_result[i] = x
                    
                    y = example.label
                    # Build one-hot representation of y. Instead of the label 0 or 1, y_onehot is either [0, 1] or [1, 0]. This
                    # way we can take the dot product directly with a probability vector to get class probabilities.
                    y_onehot = torch.zeros(num_classes)
                    # scatter will write the value of 1 into the position of y_onehot given by y
                    y_onehot.scatter_(
                        0, torch.from_numpy(np.asarray(y, dtype=np.int64)), 1
                    )
                    
                    gold_label[i] = y_onehot
                
                ffnn.zero_grad()
                log_probs = ffnn.forward(form_input(batch_result))
                gold_label = form_input(gold_label)
                
                loss = (torch.neg(torch.flatten(log_probs)).dot(torch.flatten(gold_label))) / torch.tensor(batch_size)
                
                loss.backward()
                optimizer.step()

    # No batching
    else:
        for _ in range(num_epochs):
            random.shuffle(train_exs)
            for example in train_exs:
                x_avg = avg_embeddings(example.words, word_embeddings)
                x = form_input(x_avg)
                y = example.label

                # Build one-hot representation of y. Instead of the label 0 or 1, y_onehot is either [0, 1] or [1, 0]. This
                # way we can take the dot product directly with a probability vector to get class probabilities.
                y_onehot = torch.zeros(num_classes)
                # scatter will write the value of 1 into the position of y_onehot given by y
                y_onehot.scatter_(0, torch.from_numpy(np.asarray(y, dtype=np.int64)), 1)
                # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
                ffnn.zero_grad()
                log_probs = ffnn.forward(x)
                # Can also use built-in NLLLoss as a shortcut here but we're being explicit here
                loss = torch.neg(log_probs).dot(y_onehot)
                # Computes the gradient and takes the optimizer step
                loss.backward()
                optimizer.step()

    nsc = NeuralSentimentClassifier(word_embeddings, ffnn)
    return nsc
