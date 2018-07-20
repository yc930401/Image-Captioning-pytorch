from __future__ import unicode_literals, print_function, division
import warnings
warnings.filterwarnings("ignore")

import gensim
from nltk.tokenize import WordPunctTokenizer
from PIL import Image

import os
import re
import sys
import getopt
import random
import numpy as np
from io import open
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
def cuda_variable(tensor):
    if device == 'cuda':
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


train_caption_dir = 'data/captions_train2014.txt'
val_caption_dir = 'data/captions_val2014.txt'
train_image_dir = 'F:/Dataset for Deep Learning/Image/Image Captioning-Coco2014/train2014'
val_image_dir = 'F:/Dataset for Deep Learning/Image/Image Captioning-Coco2014/val2014'


hidden_size = 256
embed_size =256
vocab_size = 0
num_layers = 1
teacher_forcing_ratio = 0.5
data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomSizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)


def prepare_vocabulary(path):
    word_tokenizor = WordPunctTokenizer()
    lines = open(path, encoding='utf-8').read().strip().split('\n')
    captions = [line.split('\t')[1] for line in lines if len(line.split('\t'))==2]
    captions = [word_tokenizor.tokenize(caption) for caption in captions]
    captions = [[word.lower() for word in caption] for caption in captions]
    length = [len(caption)+1 for caption in captions]
    dictionary = gensim.corpora.Dictionary(captions)
    #dictionary.filter_extremes(keep_n=6000)
    word_to_id = dictionary.token2id
    word_to_id['<SOS>'] = len(word_to_id)
    word_to_id['<EOS>'] = len(word_to_id)
    word_to_id['UNK'] = len(word_to_id)
    id_to_word = {id: word for word, id in word_to_id.items()}
    max_length = max(length)
    return id_to_word, word_to_id, len(id_to_word), max_length


id_to_word, word_to_id, vocab_size, max_length = prepare_vocabulary(val_caption_dir)
print('Vocab size: {}\nMax sentence length: {}'.format(vocab_size, max_length))


class CocoDataset(Dataset):

    def __init__(self, image_dir, caption_dir, n_samples=5000, transform=None):
        self.image_dir = image_dir
        self.caption_dir = caption_dir
        self.transform = transform
        self.file_names = os.listdir(self.image_dir)[:n_samples]
        self.word_tokenizor = WordPunctTokenizer()
        self.id_to_captions = {}
        for id_caption in open(caption_dir, encoding='utf-8').read().strip().split('\n'):
            if len(id_caption.split('\t')) == 2:
                id, caption = id_caption.split('\t')
                self.id_to_captions[id] = caption.lower()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.file_names[idx]))
        image = image.convert('RGB')
        caption = [word_to_id[word] if word in word_to_id.keys() else word_to_id['UNK'] for word in
                   self.word_tokenizor.tokenize(self.id_to_captions[self.file_names[idx]])]
        caption = torch.Tensor(caption + [word_to_id['<EOS>']], device = device).view(-1, 1)
        caption = caption.long()

        if self.transform:
            image_new = self.transform(image)
        sample = {'image': image_new, 'caption': caption}
        return sample


######################################################################
# The Encoder
#
# The encoder of is a pre-trained VGG-19 Network that outputs features
# of an image.
#

class EncoderCNN(nn.Module):
    def __init__(self, hidden_size):
        super(EncoderCNN, self).__init__()
        vggnet = models.vgg19(pretrained=True)
        modules = list(vggnet.features.children())[:-1]
        self.vggnet = nn.Sequential(*modules)
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv2d(512, self.hidden_size, kernel_size=1, padding=0) # return 14*14*hidden_size
        self.bn = nn.BatchNorm2d(self.hidden_size, momentum=0.01)
        self.conv2 = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=14, padding=0)


    def forward(self, images):
        features = self.vggnet(images)
        features = Variable(features.data)
        features = F.relu(self.conv1(features))
        features_for_attention = self.bn(features) # torch.Size([1, 256, 14, 14])
        features_for_initial_state = self.conv2(features_for_attention).view(1, 1, -1) # torch.Size([1, 1, 256]), combination of features in 14 locations

        return features_for_initial_state, features_for_attention



######################################################################
# Attention Decoder
#
# If only the context vector is passed betweeen the encoder and decoder,
# that single vector carries the burden of encoding the entire image.
#
# Attention allows the decoder network to "focus" on a different part of
# the encoder's outputs for every step of the decoder's own outputs. First
# we calculate a set of *attention weights*. These will be multiplied by
# the encoder output vectors to create a weighted combination.
#

class AttnDecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(self.dropout_p)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gru = nn.GRU(embed_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.init_weights()

    def init_weights(self):
        #torch.nn.init.xavier_normal(self.embedding.weight.data)
        #torch.nn.init.xavier_normal(self.linear.weight.data)
        #torch.nn.init.xavier_normal(self.linear.bias.data)
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def init_hidden(self):
        return cuda_variable(torch.zeros(self.num_layers, 1, self.hidden_size))

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(torch.bmm(encoder_outputs.view(1, -1, self.hidden_size),
                                                  hidden.view(1, self.hidden_size, 1)), dim=1)
        context_with_attn = torch.sum(torch.mul(attn_weights, encoder_outputs.view(-1, self.hidden_size)), dim=1)
        context_with_attn = context_with_attn.unsqueeze(1)
        output = torch.cat((embedded[0], context_with_attn[0]), 1)
        output = self.attn_combine(output).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.linear(output[0]), dim=1)
        return output, hidden, attn_weights


def train(encoder, decoder, dataset, batch_size, criterion, encoder_optimizer, decoder_optimizer):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    i = 0
    total_loss = []
    for i_batch, sample_batched in enumerate(dataloader):
        i += 1
        #print('Now training image No.', i)
        image, caption = sample_batched['image'].cuda(), sample_batched['caption'].cuda()[0]
        caption_length = caption.size(0)
        # Encode
        features_for_initial_state, features_for_attention = encoder(image)
        # Decode
        decoder_input = torch.tensor([[word_to_id['<SOS>']]], device=device)
        decoder_hidden = features_for_initial_state

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        loss = 0
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(caption_length):
                #print('Decoder input (truth): ', decoder_input.item(), id_to_word[decoder_input.item()])
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, features_for_attention)
                #topv, topi = decoder_output.data.topk(1)
                #print('Decoder output:, ', topi.item(), id_to_word[topi.item()])
                loss += criterion(decoder_output, caption[di])
                decoder_input = caption[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(caption_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, features_for_attention)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                loss += criterion(decoder_output, caption[di])
                if decoder_input.item() == word_to_id['<EOS>']:
                    break
        total_loss.append(loss.item()/caption_length)
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

    return total_loss


def trainIters(encoder, decoder, learning_rate, weight_decay, batch_size, n_epochs=1, n_samples=50):
    dataset = CocoDataset(val_image_dir, val_caption_dir, n_samples=n_samples, transform=data_transform)
    criterion = nn.NLLLoss()

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for i in range(n_epochs):
        total_loss = train(encoder, decoder, dataset, batch_size, criterion, encoder_optimizer, decoder_optimizer)
        loss = np.mean(total_loss)
        torch.save(encoder, 'model/encoder.pkl')
        torch.save(decoder, 'model/decoder.pkl')
        print('Iter: {}, Loss: {}, Total loss: {}'.format(i, loss, total_loss))


def evaluate(encoder, decoder, image):
    with torch.no_grad():
        features_for_initial_state, features_for_attention = encoder(image)

        decoder_input = torch.tensor([[word_to_id['<SOS>']]], device=device)
        decoder_hidden = features_for_initial_state

        decoded_words = []
        attentions = []

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, features_for_attention)
            attentions.append(decoder_attention)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == word_to_id['<EOS>']:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(id_to_word[topi.item()])
            decoder_input = topi.squeeze().detach()

        return decoded_words[:-1], attentions


def show_attention(attentions, image, file_name):
    image = Image.open(os.path.join(val_image_dir, file_name))
    image = transforms.Resize(224)(image)
    image = transforms.RandomSizedCrop(224)(image)
    #image.save('result/Original.jpg')
    caption_length = len(attentions)
    if caption_length%4 == 0:
        n_col = caption_length//4
    else:
        n_col = (caption_length//4) + 1
    plt.figure(figsize=(18, 10))
    plt.subplot(4, n_col, 1)
    plt.imshow(image)
    for i in range(len(attentions)):
        attention = attentions[i]
        attention = Image.fromarray(np.array(attention, dtype='uint8')
                                    .reshape(14, 14) * 255).resize((224, 224), Image.ANTIALIAS)
       # attention.save('result/Attention_{}.jpg'.format(i+1))
        plt.subplot(4, n_col, i+2)
        plt.imshow(attention)
    plt.show()
    plt.savefig(file_name)


def evaluateRandomly(encoder, decoder, n_samples=5):
    dataset = CocoDataset(val_image_dir, val_caption_dir, n_samples=n_samples, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    i = 0
    for i_batch, sample_batched in enumerate(dataloader):
        image, caption = sample_batched['image'].cuda(), sample_batched['caption'].cuda()[0]
        words, attentions = evaluate(encoder, decoder, image)
        output_sent = ' '.join(words)
        truth_sent = ' '.join([id_to_word[id.item()] for id in caption][:-1])
        print('Ground Truth: ', truth_sent)
        print('Output: ', output_sent)
        show_attention(attentions, image, dataset.file_names[i])
        i += 1



if __name__ == '__main__':
    try:
        encoder = torch.load('model/encoder.pkl')
        decoder = torch.load('model/decoder.pkl')
    except:
        encoder = EncoderCNN(hidden_size).to(device)
        decoder = AttnDecoderRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    # Train a model
    #trainIters(encoder, decoder , learning_rate=0.00000001, weight_decay=0.95, batch_size=1, n_epochs=3, n_samples=100)
    # evaluate a model
    evaluateRandomly(encoder, decoder, n_samples=1)


