
import torch
import os
import pathlib
import numpy as np
import glob
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import re
import string
import math
from tqdm.auto import tqdm
from collections import Counter
from torch.utils.data import DataLoader, Subset, Dataset
from torch import nn

#ggplot' ist ein spezifischer Stil, der von der R-Programmiersprache und dem ggplot2-Paket inspiriert ist. Dieser Stil erzeugt Plots mit einem grauen Hintergrund und weißen Gitterlinien
plt.style.use('ggplot')

# Set seed.
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

OUTPUTS_DIR = 'outputs'
os.makedirs(OUTPUTS_DIR, exist_ok=True)
data_dir = os.path.join('extracted_data/aclImdb/aclImdb/')
dataset_dir = os.path.join(data_dir)
train_dir = os.path.join(dataset_dir, 'train')
print(os.listdir(dataset_dir))
print(os.listdir(train_dir))

def load_data(directory):
    texts = []
    labels = []
    for label in ['pos', 'neg']:
        label_dir = os.path.join(directory, label)
        for filename in os.listdir(label_dir):
            with open(os.path.join(label_dir, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())
            labels.append(1 if label == 'pos' else 0)
    return texts, labels

# Laden der Trainingsdaten
train_texts, train_labels = load_data(train_dir)

# Laden der Testdaten
test_dir = os.path.join(dataset_dir, 'test')
test_texts, test_labels = load_data(test_dir)

print(f"Training set: {len(train_texts)} reviews")
print(f"Test set: {len(test_texts)} reviews")
print(train_texts[5])

def clean_text(text):
    # Entfernt alle Sonderzeichen und Zahlen
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Konvertiert zu Kleinbuchstaben
    text = text.lower()
    # Entfernt mehrfache Leerzeichen
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Anwendung auf die Daten
#Ich werde beides testen und dann die Ergebnisse vergleichen
train_texts = [clean_text(text) for text in train_texts]
test_texts = [clean_text(text) for text in test_texts]
print(train_texts[0])



# Teilt den Text in Wörter auf
def tokenize(text):
    # Entfernt Sonderzeichen und wandelt in Kleinbuchstaben um
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    # Teilt den Text in Wörter
    return text.split()

# Tokenisierung anwenden
train_tokens = [tokenize(text) for text in train_texts]
test_tokens = [tokenize(text) for text in test_texts]
print(train_tokens[0])

# Vokabular erstellen
# auch hier sollte ich noch verschiedene Häufigkeiten von den Wörter testen
# also verschiedene Häufigkeiten von den Wörtern in den Texten und dann die Ergebnisse vergleichen
# auch hier wird nur train_tokens verwendet und nicht test_tokens
# es kann ja sein, dass Wörter in den test_tokens anders sind als in den train_tokens
all_words = [word for text in train_tokens for word in text]
word_counts = Counter(all_words)
vocab = ['<PAD>', '<UNK>'] + [word for word, count in word_counts.items() if count > 5]
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
print(len(vocab))
print(word_to_idx)
print(word_to_idx['and'])


#weißt den einzelneen Wörtern einem Index zu
import json

# Pfad für die JSON-Datei
json_file_path = os.path.join(OUTPUTS_DIR, 'word_to_idx.json')

# Speichern des word_to_idx Wörterbuchs als JSON
with open(json_file_path, 'w') as json_file:
    json.dump(word_to_idx, json_file, indent=4)

print(f"word_to_idx wurde gespeichert in: {json_file_path}")


def tokens_to_indices(tokens):
    # Konvertiert Tokens zu Indizes und beschränkt auf MAX_LEN
    indices = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in tokens[:MAX_LEN]]
    # Fügt Padding hinzu, wenn nötig
    padding = [word_to_idx['<PAD>']] * (MAX_LEN - len(indices))
    return indices + padding

# Anwenden der Konvertierung
train_indices = [tokens_to_indices(tokens) for tokens in train_tokens]
test_indices = [tokens_to_indices(tokens) for tokens in test_tokens]
print(train_indices[0])
print(len(train_indices[0]))

print(test_indices[0])
print(len(test_indices[0]))


import json

# Pfad für die JSON-Datei
json_file_path = os.path.join(OUTPUTS_DIR, 'test_indices.json')

# Speichern des word_to_idx Wörterbuchs als JSON
with open(json_file_path, 'w') as json_file:
    json.dump(test_indices, json_file, indent=4)

print(f"test_indices wurde gespeichert in: {json_file_path}")

import json

# Pfad für die JSON-Datei
json_file_path = os.path.join(OUTPUTS_DIR, 'test_labels.json')

# Speichern des word_to_idx Wörterbuchs als JSON
with open(json_file_path, 'w') as json_file:
    json.dump(test_labels, json_file, indent=4)

print(f"test_labels wurde gespeichert in: {json_file_path}")


import json

# Pfad für die JSON-Datei
json_file_path = os.path.join(OUTPUTS_DIR, 'train_labels.json')

# Speichern des word_to_idx Wörterbuchs als JSON
with open(json_file_path, 'w') as json_file:
    json.dump(train_labels, json_file, indent=4)

print(f"train_labels wurde gespeichert in: {json_file_path}")



import json

# Pfad für die JSON-Datei
json_file_path = os.path.join(OUTPUTS_DIR, 'train_indices.json')

# Speichern des word_to_idx Wörterbuchs als JSON
with open(json_file_path, 'w') as json_file:
    json.dump(train_indices, json_file, indent=4)

print(f"train_indices wurde gespeichert in: {json_file_path}")


import json
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, reviews_path, labels_path):
        """
        Lädt die Reviews und Labels von den angegebenen JSON-Dateien.

        :param reviews_path: Dateipfad zur JSON-Datei mit den Reviews
        :param labels_path: Dateipfad zur JSON-Datei mit den Labels
        """
        # Lade die JSON-Daten von den angegebenen Pfaden
        with open(reviews_path, 'r') as f:
            self.reviews = json.load(f)
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
        
        self.len = len(self.reviews)  # Länge des Datasets

    def __len__(self):
        """Gibt die Anzahl der Beispiele im Dataset zurück."""
        return self.len

    def __getitem__(self, idx):
        """
        Gibt das Beispiel an Index `idx` zurück, konvertiert die Review und das Label in Tensoren.

        :param idx: Index des gewünschten Beispiels
        :return: Tuple (Review, Label)
        """
        review = torch.LongTensor(self.reviews[idx])  # Review als LongTensor
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Label als LongTensor
        return review, label



training_data  = CustomDataset("outputs/train_indices.json", "outputs/train_labels.json")
train_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.shape}")
print(f"Labels batch shape: {train_labels.shape}")


test_data  = CustomDataset("outputs/test_indices.json", "outputs/test_labels.json")
test_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.shape}")
print(f"Labels batch shape: {train_labels.shape}")


#muss noch gemacht werden. Bin mir nicht genau sicher wie genau und wann ich das brauche
#hab ich auch nicht wirklich verstanden
def create_mask(self, sequence):
        """
        Creates attention mask for the sequence.
        1 for actual tokens, 0 for padding tokens.
        """
        return [1 if token != self.pad_idx else 0 for token in sequence]
    
#mask = create_mask(self, sequence)


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len):
        super(Embedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.postitional_encoding = self.create_positional_encoding(max_len, embedding_dim)
        
    def create_positional_encoding(self, max_len, embedding_dim):
        positional_encoding = torch.zeros(max_len, embedding_dim)
        for pos in range(max_len):
            for i in range(embedding_dim, 2):
                positional_encoding[pos, i] = math.sin(pos / (10000 ** (( 2 * i) / embedding_dim)))
                positional_encoding[pos, i + 1] = math.cos(pos / (10000 ** (( 2* (i + 1))/ embedding_dim)))
                
        positional_encoding = positional_encoding.unsqueeze(0)  # (1, max_len, embedding_dim)
        # später wird durch Addition shape automatisch zu (batch_size, max_len, embedding_dim)
        return positional_encoding
        
    def forward(self, encoded_data):
        embeddings = self.embedding(encoded_data) * math.sqrt(self.embedding_dim) # (batch_size, max_len, embedding_dim)
        max_words = embeddings.size(1)
        embeddings = embeddings + self.postitional_encoding[:, :max_words] # positional_encoding wird automatisch auf die richtige Größe geändert
        embeddings = self.dropout(embeddings)
        return embeddings
        


class EncoderClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len, num_layers, num_heads):
        super(EncoderClassifier, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim, max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers,
        )
        self.classifier = nn.Linear(embedding_dim, 1)
        self.dropout = nn.Dropout(0.1)

    # das übergeben der maske hab ich erstmal weggelassen, kann später noch ergänzt werden
    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.dropout(x)
        x = x.max(dim=1)[0]
        out = self.linear(x)
        return out 


class AdamWarmup():
    def __init__(self, model_size, warmup_steps, optimizer):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.current_step = 0
        self.lr = 0
        
    def get_lr(self):
        return self.model_size ** (-0.5) * min(self.current_step ** (-0.5), self.current_step * self.warmup_steps ** (-1.5))
        
    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.lr = lr
        # update the weights
        self.optimizer.step()
        
        

        
    #every time you want to update the weights you call 
    #w = w - lr * grad
    #optimizer.step()
    


model = EncoderClassifier(len(vocab), EMBED_DIM, MAX_LEN, NUM_ENCODER_LAYERS, NUM_HEADS).to(DEVICE)
print(model)
adam_opimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
transformer_optimizer = AdamWarmup(model_size=EMBED_DIM, warmup_steps=4000, optimizer=adam_opimizer)
criterion = nn.BCEWithLogitsLoss()




def train(model, train_loader, criterion, epoch):
    model.train()
    print("Training ist starting...")
    

    counter = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(DEVICE), 
        target = target.to(DEVICE)
        
        #Run the model
        output = model(data)
        
        #Der Aufruf torch.squeeze(outputs, -1) sorgt also dafür, dass die Ausgabe des Modells dieselbe Dimension wie die Labels hat und für die Berechnung des Loss vorbereitet ist.
        outputs = torch.squeeze(outputs, -1)
        loss = criterion(output, target)
        
        #Backpropagation
        adam_opimizer.optimizer.zero_grad()
        loss.backward()
        adam_opimizer.step()
        
        samples = data.shape[0]
        
        sum_loss += loss.item() * samples
        counter += samples
        
        #Print the results
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), sum_loss / counter))
        
        


#lass ich beim Training erstmal weg. Kann später noch einmal probieren
def validate(model, test_loader, criterion, device):
    model.eval()
    print("Evaluating...")
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for data, target in test_loader:
            counter += 1
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            outputs = torch.squeeze(outputs, -1)

            test_loss = criterion(outputs, target)
            valid_running_loss += test_loss.item()
            valid_running_correct += count_correct_incorrect(target, outputs, valid_running_correct)
            
    print("Validation loss: {:.4f}".format(valid_running_loss / counter))
    print("Validation accuracy: {:.4f}".format( 100. * valid_running_correct / len(test_loader.dataset)))
    
def count_correct_incorrect(labels, outputs, train_running_correct):
    # As the outputs are currently logits.
    outputs = torch.sigmoid(outputs)
    running_correct = 0
    for i, label in enumerate(labels):
        if label < 0.5 and outputs[i] < 0.5:
            running_correct += 1
        elif label >= 0.5 and outputs[i] >= 0.5:
            running_correct += 1
    return running_correct

for epoch in range(EPOCHS):
    print(f"[INFO]: Epoch {epoch+1} of {EPOCHS}")
    train( model=model,train_loader=train_loader, criterion=criterion, epoch=epoch)
    validate( model=model, val_loader=test_loader, criterion=criterion, epoch=epoch)
    torch.save(model.state_dict(), f"models/transformer_classification_{epoch+1}.pt")
    



