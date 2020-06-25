import dill
import torch
from nltk.tokenize import TweetTokenizer

from application.models.model import SentimentRNN

# setup device to cpu
device = torch.device('cpu')

# Load source and destination field
with open("./application/models/brain/vacab_to_int.dict", "rb")as f:
    vocab_to_int = dill.load(f)

vocab_size = len(vocab_to_int) + 1  # +1 for the 0 padding + our word tokens
output_size = 1
embedding_dim = 512
hidden_dim = 100
n_layers = 2

model = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

model.load_state_dict(torch.load('./application/models/brain/model.brain', map_location=device))
model.eval()

tokenizer = TweetTokenizer()


def predict(sentence):
    tokenized = tokenizer.tokenize(sentence)
    input_ = torch.tensor([[vocab_to_int[word] for word in tokenized]])
    hidden = model.init_hidden(1)
    prediction = model.forward(input_, hidden)[0].tolist()[0]
    return prediction
