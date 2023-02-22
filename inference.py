import torch
import spacy

def inference(model, vocab, input_text, max_len=15):
    model.eval()
    # use spacy to tokenize the input text
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(input_text)
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    # convert tokens to indices
    indices = [vocab[token] for token in tokens]
    # add <SOS> to the start of the indices 
    indices = [vocab['<SOS>']] + indices
    
    """Since we don't have access to the actual target output sequence during inference, 
    we use a "fake" label tensor. This fake label tensor contains a single "<EOS>" token,
    followed by padding tokens."""
    label = [vocab['<EOS>']] + [vocab['<PAD>']] * (max_len - 1)
    label = torch.tensor(label)

    # convert the indices to a tensor
    indices = torch.tensor(indices)
    # add a batch dimension
    indices = indices.unsqueeze(0)
    label = label.unsqueeze(0)
    # initialize the output sequence
    output_seq = []
    # loop through the indices
    for i in range(max_len):
        # get the next input token
        input = indices[:, i]
        # get the next target token
        target = label[:, i]
        # pass the input and target to the model
        print(input, target)
        output = model(input, target)
        # get the index of the predicted token
        pred = output.argmax(1)
        # add the predicted token to the output sequence
        output_seq.append(pred.item())
        # update the input sequence
        indices[:, i + 1] = pred
    # convert the output sequence to a string
    output_text = [vocab.itos[idx] for idx in output_seq]
    output_text = ' '.join(output_text)
    return output_text





    