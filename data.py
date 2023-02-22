import torch
from torch.utils.data import Dataset
import os
import spacy
import pickle


class EmailDataset(Dataset):

    def __init__(self,
                 file_path,
                 seq_len=15,
                 label_len=3,
                 vocab_pickle='vocab.pkl'):
        self.seq_len = seq_len
        self.label_len = label_len
        self.vocab_pickle = vocab_pickle
        self.vocab = self.build_vocab(file_path)
        self.data = self.process_file(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def build_vocab(self, file_path):
        # check if vocab pickle exists
        if os.path.exists(self.vocab_pickle):
            print('Loading vocab from pickle file')
            with open(self.vocab_pickle, 'rb') as f:
                vocab = pickle.load(f)
            return vocab
        else:
            print('Building vocab from scratch')
            # create vocab pickle

            with open(file_path, 'r') as f:
                text = f.read()

            nlp = spacy.load('en_core_web_sm')
            nlp.max_length = 592506583
            doc = nlp(text)

            # Create a dictionary of words and their counts
            word_counts = {'<PAD>': 1, '<SOS>': 1, '<EOS>': 1}
            for token in doc:
                if token.is_alpha:
                    word = token.text.lower()
                    word_counts[word] = word_counts.get(word, 0) + 1

            # Sort words by frequency and assign a unique index to each
            sorted_words = sorted(word_counts.items(),
                                  key=lambda x: x[1],
                                  reverse=True)
            vocab = {
                word: idx
                for idx, (word, count) in enumerate(sorted_words)
            }
            # Save the vocab to a pickle file
            with open(self.vocab_pickle, 'wb') as f:
                pickle.dump(vocab, f)
        print('Vocab built and saved to {}'.format(self.vocab_pickle))
        return vocab

    def process_file(self, file_path):
        print('Processing file {}'.format(file_path))
        with open(file_path, 'r') as f:
            text = f.read()

        nlp = spacy.load('en_core_web_sm')
        nlp.max_length = 592506583
        doc = nlp(text)
        vocab = self.vocab

        # Create a list of all word indices in the text
        word_indices = [
            vocab[token.text.lower()] for token in doc if token.is_alpha
        ]

        # Add padding tokens to the start and end of the list
        padded_indices = [vocab['<PAD>']] * (
            self.seq_len + self.label_len) + word_indices + [vocab['<PAD>']
                                                             ] * self.label_len

        # Split the list into input sequences and labels
        data = []
        for i in range(self.seq_len,
                       len(padded_indices) - self.label_len, self.label_len):
            # add <SOS> and <EOS> tokens to the input sequence
            seq = [vocab['<SOS>']
                   ] + padded_indices[i - self.seq_len:i] + [vocab['<EOS>']]
            label = [vocab['<SOS>']] + padded_indices[i:i + self.label_len] + [
                vocab['<EOS>']
            ]
            data.append((torch.tensor(seq), torch.tensor(label)))
        print('File processed')
        return data
