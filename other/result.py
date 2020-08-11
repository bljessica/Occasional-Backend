# coding: utf-8

# In[40]:


import os;
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "upload_pictures.settings")  # NoQA
import django;
django.setup()  # NoQA


import numpy as np
# from string import punctuation
from collections import Counter
import re, wikipedia, torch
import json
import os
from watson_developer_cloud import VisualRecognitionV3


def data_prepare(f1, f2):
    with open(f1, 'r', encoding='UTF-8') as f:
        reviews = f.read()
    with open(f2, 'r', encoding='UTF-8') as f:
        labels = f.read()
    reviews = reviews.lower()
    all_text = ''.join([c for c in reviews if c not in punctuation])
    reviews_split = all_text.split('\n')
    all_text = ' '.join(reviews_split)
    words = all_text.split()
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    reviews_ints = []
    for review in reviews_split:
        reviews_ints.append([vocab_to_int[word] for word in review.split()])
    labels_split = labels.split('\n')
    encoded_labels = []
    for label in labels_split[0]:
        if label == "0":
            encoded_labels.append(0)
        if label == "1":
            encoded_labels.append(1)
        if label == "2":
            encoded_labels.append(2)
        if label == "3":
            encoded_labels.append(3)
    non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]
    reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
    encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])
    return reviews_ints, encoded_labels, vocab_to_int


# In[41]:


visual_recognition = VisualRecognitionV3(
    '2018-03-19',
    iam_apikey='XfY9pIiwDUiWbwH28wjBulIjykUaNgr8f9HDXuN_d6L5')

# In[42]:

path = os.getcwd()
text_path = path + '/other/wastetext.txt'
label_path = path + '/other/wastelabel.txt'
pth_path = path + '/other/text_rnn.pth'
reviews_ints, encoded_labels, vocab_to_int = data_prepare(text_path,
                                                          label_path)

# In[43]:


encoded_labels.reshape(1, -1)
while len(reviews_ints) > 25000:
    del reviews_ints[25000]
    encoded_labels = np.delete(encoded_labels, 25000, 0)


# In[44]:


def pad_features(reviews_ints, seq_length):
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    return features


# In[45]:

seq_length = 100
features = pad_features(reviews_ints, seq_length=seq_length)

# In[46]:


# First checking if GPU is available
train_on_gpu = torch.cuda.is_available()
if (train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

# In[47]:


import torch
import torch.nn as nn
import torch.nn.functional as F


class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim, output_size)
        self.fc2 = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = F.relu(self.fc1(out))
        out = out.view(batch_size, seq_length, output_size)
        out = out[:, -1]
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden


vocab_size = len(vocab_to_int) + 1  # +1 for the 0 padding + our word tokens
output_size = 4
embedding_dim = 200
hidden_dim = 128
n_layers = 2
net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
lr = 0.005
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
net.load_state_dict(torch.load(pth_path))


# In[48]:


def predict(net, test_review, sequence_length):
    classes = ['wet', 'dry', 'recycable', 'toxic']
    net.eval()
    test_ints = tokenize_review(test_review)
    seq_length = sequence_length
    features = pad_features(test_ints, seq_length)
    feature_tensor = torch.from_numpy(features)
    batch_size = feature_tensor.size(0)
    h = net.init_hidden(batch_size)
    if (train_on_gpu):
        feature_tensor = feature_tensor.cuda()
    output, h = net(feature_tensor, h)
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())
    return (classes[preds])


# In[49]:


from string import punctuation


def tokenize_review(test_review):
    test_review = test_review.lower()
    test_text = ''.join([c for c in test_review if c not in punctuation])
    test_words = test_text.split()
    test_ints = []
    for word in test_words:
        try:
            test_ints.append(vocab_to_int[word])
        except:
            continue
    test_intss = []
    test_intss.append(test_ints)
    return test_intss


# In[52]:


def waste_classification(inputs, seq_length):
    if re.match(r'^http?:/{2}\w.+$', inputs):
        url = inputs
        classes_result = visual_recognition.classify(url=url).get_result()
        s = classes_result["images"][0]["classifiers"][0]["classes"][0]["class"]
        print(s)
        s = re.sub('[^a-zA-Z]', ' ', s)
        temp = s.split(" ")
        test_review = ""
        if len(temp) > 1:
            try:
                test_review = test_review + " " + wikipedia.summary(s, sentences=1)
            except Exception as e:
                e = str(e)
                e = e[e.index("\n") + 1:]
                test_review = test_review + " " + wikipedia.summary(e[:e.index("\n")], sentences=1)
        for x in temp:
            try:
                test_review = test_review + " " + wikipedia.summary(x, sentences=1)
            except Exception as e:
                e = str(e)
                e = e[e.index("\n") + 1:]
                test_review = test_review + " " + wikipedia.summary(e[:e.index("\n")], sentences=1)
        test_ints = tokenize_review(test_review)
        features = pad_features(test_ints, seq_length)
        feature_tensor = torch.from_numpy(features)
        return s, predict(net, test_review, seq_length)
    else:
        s = inputs
        s = re.sub('[^a-zA-Z]', ' ', s)
        temp = s.split(" ")
        test_review = ""
        if len(temp) > 1:
            try:
                test_review = test_review + " " + wikipedia.summary(s, sentences=1)
            except Exception as e:
                e = str(e)
                e = e[e.index("\n") + 1:]
                test_review = test_review + " " + wikipedia.summary(e[:e.index("\n")], sentences=1)
        for x in temp:
            try:
                test_review = test_review + " " + wikipedia.summary(x, sentences=1)
            except Exception as e:
                e = str(e)
                e = e[e.index("\n") + 1:]
                test_review = test_review + " " + wikipedia.summary(e[:e.index("\n")], sentences=1)
        test_ints = tokenize_review(test_review)
        features = pad_features(test_ints, seq_length)
        feature_tensor = torch.from_numpy(features)
        return inputs, predict(net, test_review, seq_length)
