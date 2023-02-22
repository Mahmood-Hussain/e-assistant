import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import EmailDataset
from model import Encoder, Decoder, Seq2Seq
import time
import math
import argparse
import os


def train(model, iterator, optimizer, criterion, clip=1):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src, trg = batch
        src = src.to(model.device)
        trg = trg.to(model.device)
        src = src.permute(1, 0)
        trg = trg.permute(1, 0)

        optimizer.zero_grad()

        output = model(src, trg)

        # Truncate the last token in the output and the first token in the target
        # since we do not need to calculate loss for <SOS> token in output and <EOS> token in target
        output = output[:-1, :]
        trg = trg[1:, :]

        # Reshape the output and target to 2D tensor for calculating loss
        output = output.reshape(-1, output.shape[-1])
        trg = trg.reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return model, epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src, trg = batch
            src = src.to(model.device)
            trg = trg.to(model.device)
            src = src.permute(1, 0)
            trg = trg.permute(1, 0)

            output = model(src, trg, 0)  #turn off teacher forcing

            output = output[:-1, :]
            trg = trg[1:, :]

            # Reshape the output and target to 2D tensor for calculating loss
            output = output.reshape(-1, output.shape[-1])
            trg = trg.reshape(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def inference(model, vocab, src, max_len=3, device='cpu'):
    model.eval()
    with torch.no_grad():
        src = torch.LongTensor(src).unsqueeze(1).to(device)
        hidden, cell = model.encoder(src)
        outputs = [vocab['<SOS>']]
        for i in range(max_len):
            previous_word = torch.LongTensor([outputs[-1]]).to(device)
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()
            outputs.append(best_guess)
            if best_guess == vocab['<EOS>']:
                break
        # get words back from outputs indices
        output_strings = []
        for token in outputs:
            tokens_with_same_value = [
                k for k, v in vocab.items() if v == token
            ]
            output_string = ' '.join(tokens_with_same_value)
            # remove <SOS> and <EOS> tokens
            if output_string not in ['<SOS>', '<EOS>']:
                output_strings.append(output_string)

        output_text = ' '.join(output_strings)

        return output_text


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# get all above as arguments make above values default
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seq_len', type=int, default=15)
parser.add_argument('--label_len', type=int, default=3)
parser.add_argument('--embedding_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--checkpint_dir', type=str, default='checkpoints')
parser.add_argument('--save_epoch', type=int, default=5)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--text_file_path',
                    type=str,
                    default='data/cleaned_emails.txt_0.001.txt')
parser.add_argument('--inference_mode', type=bool, default=False)
parser.add_argument('--model_path', type=str, default='checkpoints/model.pt')
parser.add_argument('--inference_text', type=str, default='Hi')
args = parser.parse_args()

# get arguments
num_epochs = args.num_epochs
learning_rate = args.learning_rate
batch_size = args.batch_size
seq_len = args.seq_len
label_len = args.label_len
embedding_size = args.embedding_size
hidden_size = args.hidden_size
num_layers = args.num_layers
dropout = args.dropout
save_epoch = args.save_epoch
checkpoint_dir = args.checkpint_dir
device = 'cpu' if args.device == 'cpu' else torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')
text_file_path = args.text_file_path
inference_mode = args.inference_mode
model_path = args.model_path
inference_text = args.inference_text

print('device: ', device)

# Define dataset and data loader
email_dataset = EmailDataset(file_path=text_file_path,
                             seq_len=seq_len,
                             label_len=label_len)
email_loader = DataLoader(email_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, optimizer, and loss function
encoder = Encoder(len(email_dataset.vocab), embedding_size, hidden_size,
                  num_layers, dropout)
decoder = Decoder(len(email_dataset.vocab), embedding_size, hidden_size,
                  num_layers, dropout)
seq2seq_model = Seq2Seq(encoder, decoder, device).to(device)
optimizer = torch.optim.Adam(seq2seq_model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=email_dataset.vocab['<PAD>'])


CLIP = 1

best_valid_loss = float('inf')

if inference_mode == False:
    print('Training model...')
    for epoch in range(num_epochs):

        start_time = time.time()

        model, train_loss = train(seq2seq_model, email_loader, optimizer,
                                  criterion, CLIP)
        # valid_loss = evaluate(seq2seq_model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # if valid_loss < best_valid_loss:
        #     best_valid_loss = valid_loss
        #     torch.save(model.state_dict(), 'model.pt')

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}'
        )
        # print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        # save model
        if (epoch + 1) % save_epoch == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.checkpint_dir, f'model_{epoch+1}.pt'))
else:
    # inference
    assert os.path.exists(model_path), 'model path does not exist'
    checkpoint = torch.load(model_path)
    seq2seq_model.load_state_dict()
    seq2seq_model.eval()
    src = inference_text
    src = [email_dataset.vocab[token.lower()] for token in src.split()]
    # add <SOS> and <EOS> tokens
    src = [email_dataset.vocab['<SOS>']] + src + [email_dataset.vocab['<EOS>']]
    print(
        inference(seq2seq_model,
                  email_dataset.vocab,
                  src,
                  max_len=50,
                  device=device))
