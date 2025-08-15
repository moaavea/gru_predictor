
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

# Toy dataset
sentences = [
    "the cat sat on the mat",
    "the dog barked at the cat",
    "the mat was blue",
    "the dog chased the cat",
    "the cat ran away",
    "the dog sat on the rug",
    "the rug was soft",
    "the cat is sleeping",
    "the dog is barking",
    "the mat is dirty"
]

# Tokenization and vocabulary
tokenized = [s.split() for s in sentences]
word2idx = defaultdict(lambda: len(word2idx))
word2idx["<PAD>"]
word2idx["<UNK>"]
for sentence in tokenized:
    for word in sentence:
        word2idx[word]
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(word2idx)

# Convert to index sequences
sequences = []
for sentence in tokenized:
    idxs = [word2idx[word] for word in sentence]
    for i in range(1, len(idxs)):
        input_seq = idxs[:i]
        target_word = idxs[i]
        sequences.append((input_seq, target_word))

# Padding and batching
inputs = [torch.tensor(s[0]) for s in sequences]
targets = [torch.tensor(s[1]) for s in sequences]
inputs_padded = pad_sequence(inputs, batch_first=True)
targets = torch.stack(targets)

# Define GRU model
class GRUNextWordModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.gru(x)
        out = self.fc(h.squeeze(0))
        return out

model = GRUNextWordModel(vocab_size, embed_dim=32, hidden_dim=64)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
print("Training...\n")
for epoch in range(10):
    total_loss = 0
    for i in range(len(inputs_padded)):
        input_tensor = inputs_padded[i].unsqueeze(0)
        target_tensor = targets[i].unsqueeze(0)

        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/10, Loss: {total_loss:.4f}")

# Prediction function
def predict_next_word(model, text):
    model.eval()
    with torch.no_grad():
        input_idxs = [word2idx.get(w, word2idx["<UNK>"]) for w in text.split()]
        input_tensor = torch.tensor(input_idxs).unsqueeze(0)
        output = model(input_tensor)
        predicted_idx = output.argmax(dim=1).item()
        return idx2word[predicted_idx]

# Example prediction
seed_text = "the cat"
predicted_word = predict_next_word(model, seed_text)
print(f"\nPrediction after '{seed_text}': {predicted_word}")
