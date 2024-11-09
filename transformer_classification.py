#!/usr/bin/env python
# coding: utf-8

# # Erklärung der Imports
# 
# - `torch`: PyTorch-Bibliothek für Deep Learning
# - `transformers`: Hugging Face Transformers-Bibliothek für State-of-the-Art NLP-Modelle
# - `datasets`: Hugging Face Datasets-Bibliothek für einfachen Zugriff auf NLP-Datensätze
# - `numpy`: Numerische Berechnungen und Array-Operationen
# - `pandas`: Datenmanipulation und -analyse
# - `matplotlib`: Plotting und Visualisierung
# - `sklearn`: Scikit-learn für Machine Learning-Algorithmen und Metriken

# In[156]:


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


# # Seed-Einstellung für Reproduzierbarkeit
# 
# Dieser Code-Abschnitt stellt die Reproduzierbarkeit der Ergebnisse sicher, indem er die Zufallszahlengeneratoren für verschiedene Komponenten auf einen festen Wert setzt.
# 
# ## Detaillierte Erklärung
# 
# 1. **Seed-Definition**
#    - Definiert eine Seed-Zahl. 42 ist eine beliebte Wahl, aber jede Zahl könnte verwendet werden.
# 
# 2. **NumPy Seed**
#    - Setzt den Seed für NumPy's Zufallszahlengenerator.
# 
# 3. **PyTorch CPU Seed**
#    - Setzt den Seed für PyTorch's CPU-Zufallszahlengenerator.
# 
# 4. **PyTorch GPU Seed**
#    - Setzt den Seed für PyTorch's GPU-Zufallszahlengenerator.
# 
# 5. **cuDNN Determinismus**
#    - Sorgt dafür, dass cuDNN deterministische Algorithmen verwendet.
# 
# 6. **cuDNN Benchmark**
#    - Ermöglicht cuDNN, die optimale Algorithmus-Implementierung für die gegebene Hardware zu wählen.
# 
# ## Hinweis
# 
# Die letzte Einstellung (cuDNN Benchmark) kann die vollständige Reproduzierbarkeit über verschiedene Hardware-Konfigurationen hinweg beeinträchtigen. Wenn absolute Reproduzierbarkeit wichtiger ist als Leistung, sollte diese Option möglicherweise deaktiviert werden.

# In[157]:


# Set seed.
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


# # Verzeichnisstruktur und Datenzugriff
# 
# Dieser Code-Abschnitt richtet die Verzeichnisstruktur für das Projekt ein und zeigt den Inhalt wichtiger Verzeichnisse an.
# 
# 
# ## Hinweis
# 
# Dieser Code geht davon aus, dass der IMDB-Datensatz bereits heruntergeladen und in einem Verzeichnis namens 'aclImdb' im aktuellen Arbeitsverzeichnis gespeichert ist. Die Ausgabe der letzten beiden Zeilen wird die Struktur und den Inhalt des Datensatzes anzeigen.

# In[158]:


OUTPUTS_DIR = 'outputs'
os.makedirs(OUTPUTS_DIR, exist_ok=True)
data_dir = os.path.join('data/aclImdb/aclImdb/')
dataset_dir = os.path.join(data_dir)
train_dir = os.path.join(dataset_dir, 'train')
print(os.listdir(dataset_dir))
print(os.listdir(train_dir))


# # Wichtige Konfigurationsvariablen
# 
# ## Detaillierte Erklärung
# 
# 1. **MAX_LEN = 1024**
#    - Maximale Länge der Eingabesequenzen in Token.
# 
# 2. **NUM_WORDS = 32000**
#    - Größe des Vokabulars. Es werden die 32.000 häufigsten Wörter verwendet.
# 
# 3. **BATCH_SIZE = 32**
#    - Anzahl der Beispiele, die in einem Trainingsschritt verarbeitet werden.
# 
# 4. **VALID_SPLIT = 0.20**
#    - 20% der Trainingsdaten werden für die Validierung verwendet.
# 
# 5. **EPOCHS = 30**
#    - Anzahl der vollständigen Durchläufe durch den Trainingsdatensatz.
# 
# 6. **LR = 0.00001**
#    - Lernrate für den Optimierungsalgorithmus (sehr kleine Schritte).
# 
# Diese Variablen steuern wichtige Aspekte des Modelltrainings und der Datenverarbeitung. Sie können angepasst werden, um die Leistung des Modells zu optimieren oder an unterschiedliche Hardwareressourcen anzupassen.

# In[159]:


MAX_LEN = 1024
# Use these many top words from the dataset. If -1, use all words.
NUM_WORDS = 32000 # Vocabulary size.
# Batch size.
BATCH_SIZE = 32
VALID_SPLIT = 0.20
EPOCHS = 100
LR = 0.00001


# Model parameters.
EMBED_DIM = 256
NUM_ENCODER_LAYERS = 3
NUM_HEADS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)


# # Daten Laden und Verarbeiten
# 
# ## Funktion `load_data(directory)`
# 
# Diese Funktion lädt Textdaten und ihre Bezeichnungen aus einem gegebenen Verzeichnis.
# 
# ### Prozess:
# 1. Durchsucht 'pos' und 'neg' Unterverzeichnisse
# 2. Liest jede Textdatei
# 3. Speichert Texte und entsprechende Bezeichnungen (1 für positiv, 0 für negativ)

# In[160]:


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


# # Sonderzeichen und Punkte bei Transformer-Training
# 
# ## Vorteile der Entfernung:
# 
# 1. **Reduzierte Vokabulargröße**: 
#    - Weniger einzigartige Token
#    - Potenziell schnelleres Training
# 
# 2. **Fokus auf Wortbedeutung**: 
#    - Modell konzentriert sich mehr auf Wörter statt Interpunktion
# 
# 3. **Konsistenz**: 
#    - Gleiche Wörter werden immer gleich behandelt
# 
# ## Nachteile der Entfernung:
# 
# 1. **Verlust von Informationen**: 
#    - Interpunktion kann bedeutungstragend sein
#    - Satzstruktur geht verloren
# 
# 2. **Einschränkung der Modellfähigkeiten**: 
#    - Moderne Transformer können oft mit Interpunktion umgehen
#    - Entfernung könnte Modell-Leistung einschränken
# 
# 3. **Weniger natürliche Sprache**: 
#    - Text ohne Interpunktion ist weniger repräsentativ für echte Sprache
# 
# ## Empfehlung:
# 
# 1. **Experimentieren**: 
#    - Testen Sie beide Ansätze und vergleichen Sie die Ergebnisse
# 
# 2. **Aufgabenspezifische Entscheidung**: 
#    - Für reine Sentiment-Analyse könnte Entfernung okay sein
#    - Für komplexere Aufgaben besser beibehalten
# 
# 3. **Minimale Vorverarbeitung**: 
#    - Moderne Transformer profitieren oft von minimaler Vorverarbeitung
# 
# 4. **Konsistenz**: 
#    - Gleiche Vorverarbeitung für Training und Inferenz

# In[161]:


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



# In[162]:


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


# ### Erklärung des Codes
# 
# 1. **Wörter sammeln**: Erstellt eine Liste, die alle Wörter aus den Token-Listen in den Trainingsdaten enthält.
# 
# 2. **Wortzähldaten erstellen**: Erstellt ein Wörterbuch, das jedes Wort mit seiner Häufigkeit speichert.
# 
# 3. **Vokabular aufbauen**: Baut ein Vokabular auf, das mit `<PAD>` und `<UNK>` beginnt und nur Wörter enthält, die öfter als einmal vorkommen. Dadurch werden seltene Wörter ausgeschlossen.
# 
# 4. **Wort-zu-Index-Mapping**: Erstellt ein Wörterbuch, das jedem Wort im Vokabular einen eindeutigen Index zuordnet.
# 

# # Start- und End-Token bei Klassifikationsaufgaben
# 
# ## Nicht notwendig für:
# 1. Sentiment-Analyse
# 2. Textklassifikation (z.B. Themen- oder Genreklassifikation)
# 3. Andere Aufgaben, bei denen eine einzelne Klassifikation für den gesamten Text vorgenommen wird
# 
# ## Gründe:
# 1. Klassifikationsmodelle betrachten den gesamten Text als eine Einheit.
# 2. Die Reihenfolge der Wörter ist wichtig, aber Start und Ende haben keine besondere Bedeutung.
# 3. Das Modell muss keine Sequenz generieren oder den Anfang/Ende einer Sequenz erkennen.
# 
# ## Wann sind sie nützlich?
# 1. Sequenz-zu-Sequenz-Aufgaben (z.B. Übersetzung, Zusammenfassung)
# 2. Textgenerierung
# 3. Aufgaben, bei denen die Länge der Eingabe oder Ausgabe variabel ist und erkannt werden muss
# 
# ## Alternative für Klassifikation:
# - Stattdessen können Sie ein spezielles Padding-Token verwenden, um alle Eingaben auf die gleiche Länge zu bringen.

# In[163]:


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


# In[164]:


#weißt den einzelneen Wörtern einem Index zu
import json

# Pfad für die JSON-Datei
json_file_path = os.path.join(OUTPUTS_DIR, 'word_to_idx.json')

# Speichern des word_to_idx Wörterbuchs als JSON
with open(json_file_path, 'w') as json_file:
    json.dump(word_to_idx, json_file, indent=4)

print(f"word_to_idx wurde gespeichert in: {json_file_path}")


# ### Erklärung des Codes
# 
# - **Funktion**: `tokens_to_indices` wandelt Tokens in Indizes um und fügt Padding hinzu.
#   
# - **Umwandlung der Tokens in Indizes**:
#   - Jedes Wort wird in seinen numerischen Index umgewandelt.
#   - Bei unbekannten Wörtern wird der Index für `<UNK>` verwendet.
#   - Die Anzahl der Tokens wird auf eine maximale Länge (`MAX_LEN`) begrenzt.
# 
# - **Padding hinzufügen**:
#   - Wenn die Liste der Indizes kürzer als `MAX_LEN` ist, werden `<PAD>`-Indizes hinzugefügt.
#   
# - **Rückgabe**:
#   - Die Funktion gibt die vollständige Liste der Indizes zurück, ergänzt durch das erforderliche Padding, sodass alle Ausgaben eine einheitliche Länge haben.
# 

# In[165]:


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


# In[166]:


import json

# Pfad für die JSON-Datei
json_file_path = os.path.join(OUTPUTS_DIR, 'test_indices.json')

# Speichern des word_to_idx Wörterbuchs als JSON
with open(json_file_path, 'w') as json_file:
    json.dump(test_indices, json_file, indent=4)

print(f"test_indices wurde gespeichert in: {json_file_path}")


# In[167]:


import json

# Pfad für die JSON-Datei
json_file_path = os.path.join(OUTPUTS_DIR, 'test_labels.json')

# Speichern des word_to_idx Wörterbuchs als JSON
with open(json_file_path, 'w') as json_file:
    json.dump(test_labels, json_file, indent=4)

print(f"test_labels wurde gespeichert in: {json_file_path}")


# In[168]:


import json

# Pfad für die JSON-Datei
json_file_path = os.path.join(OUTPUTS_DIR, 'train_labels.json')

# Speichern des word_to_idx Wörterbuchs als JSON
with open(json_file_path, 'w') as json_file:
    json.dump(train_labels, json_file, indent=4)

print(f"train_labels wurde gespeichert in: {json_file_path}")


# In[169]:


import json

# Pfad für die JSON-Datei
json_file_path = os.path.join(OUTPUTS_DIR, 'train_indices.json')

# Speichern des word_to_idx Wörterbuchs als JSON
with open(json_file_path, 'w') as json_file:
    json.dump(train_indices, json_file, indent=4)

print(f"train_indices wurde gespeichert in: {json_file_path}")


# # `Dataset`-Klasse in PyTorch
# 
# - **Zweck:** 
#   - Die `Dataset`-Klasse in PyTorch ist eine abstrakte Klasse, die als Basis für Datensatzklassen dient. Sie ermöglicht das einfache Laden und Verarbeiten von Daten für maschinelles Lernen.
# 
# - **Erweiterung:** 
#   - Durch das Erben von `Dataset` können Entwickler ihre eigenen Datensatzklassen erstellen, die spezifisch auf die Anforderungen ihrer Anwendung abgestimmt sind.
# 
# - **Methoden:**
#   - **`__len__`**: Diese Methode gibt die Anzahl der Elemente im Datensatz zurück. Sie ermöglicht es, die Größe des Datensatzes zu bestimmen, was für die Erstellung von Mini-Batches wichtig ist.
#   
#   - **`__getitem__`**: Diese Methode ermöglicht den Zugriff auf die einzelnen Elemente im Datensatz. Sie nimmt einen Index als Parameter und gibt das entsprechende Datenpaar (z.B. Eingabe und Label) zurück. Dies ist besonders nützlich für die Iteration über den Datensatz in einem Trainingsprozess.
# 
# - **Integration mit DataLoader:**
#   - Die `Dataset`-Klasse arbeitet nahtlos mit der `DataLoader`-Klasse zusammen, die das Laden von Daten in Batches, das Mischen der Daten und die parallele Verarbeitung ermöglicht. Dies verbessert die Effizienz beim Training von Modellen.
# 
# - **Flexibilität:**
#   - Die Anpassung der `Dataset`-Klasse erlaubt es Entwicklern, Daten aus verschiedenen Quellen (wie Dateien, Datenbanken oder APIs) zu laden und sie in einem einheitlichen Format bereitzustellen, was die Wiederverwendbarkeit und Lesbarkeit des Codes verbessert.
# 
# 

# In[170]:


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


# In[171]:


training_data  = CustomDataset("outputs/train_indices.json", "outputs/train_labels.json")
train_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.shape}")
print(f"Labels batch shape: {train_labels.shape}")


test_data  = CustomDataset("outputs/test_indices.json", "outputs/test_labels.json")
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.shape}")
print(f"Labels batch shape: {train_labels.shape}")


# In[172]:


#muss noch gemacht werden. Bin mir nicht genau sicher wie genau und wann ich das brauche
#hab ich auch nicht wirklich verstanden
def create_mask(self, sequence):
        """
        Creates attention mask for the sequence.
        1 for actual tokens, 0 for padding tokens.
        """
        return [1 if token != self.pad_idx else 0 for token in sequence]
    
#mask = create_mask(self, sequence)


# In[173]:


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len):
        super(Embedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(0.1)
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.postitional_encoding = self.create_positional_encoding(max_len, embedding_dim).to(DEVICE)
        
    def create_positional_encoding(self, max_len, embedding_dim):
        positional_encoding = torch.zeros(max_len, embedding_dim)
        for pos in range(max_len):
            for i in range(0,embedding_dim, 2):
                positional_encoding[pos, i] = math.sin(pos / (10000 ** (( 2 * i) / embedding_dim)))
                positional_encoding[pos, i + 1] = math.cos(pos / (10000 ** (( 2* (i + 1))/ embedding_dim)))
                
        positional_encoding = positional_encoding.unsqueeze(0)  # (1, max_len, embedding_dim)
        # später wird durch Addition shape automatisch zu (batch_size, max_len, embedding_dim)
        return positional_encoding
        
    def forward(self, encoded_data):
        embeddings = self.embedding(encoded_data) * math.sqrt(self.embedding_dim) # (batch_size, max_len, embedding_dim)
        embeddings = embeddings.to(DEVICE)
        max_words = embeddings.size(1)
        embeddings = embeddings + self.postitional_encoding[:, :max_words] # positional_encoding wird automatisch auf die richtige Größe geändert
        embeddings = self.dropout(embeddings)
        return embeddings
        


# In[174]:


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
        out = self.classifier(x)
        return out 


# In[175]:


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
    


# In[176]:


model = EncoderClassifier(len(vocab), EMBED_DIM, MAX_LEN, NUM_ENCODER_LAYERS, NUM_HEADS).to(DEVICE)
print(model)
adam_opimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
transformer_optimizer = AdamWarmup(model_size=EMBED_DIM, warmup_steps=4000, optimizer=adam_opimizer)
criterion = nn.BCEWithLogitsLoss()



# In[181]:


def train(model, train_loader, criterion, epoch):
    model.train()
    print("Training ist starting...")
    
    sum_loss = 0
    counter = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        
        #Run the model
        outputs = model(data).to(DEVICE)
        
        #Der Aufruf torch.squeeze(outputs, -1) sorgt also dafür, dass die Ausgabe des Modells dieselbe Dimension wie die Labels hat und für die Berechnung des Loss vorbereitet ist.
        outputs = torch.squeeze(outputs, -1)
        loss = criterion(outputs, target.float())
        
        #Backpropagation
        transformer_optimizer.optimizer.zero_grad()
        loss.backward()
        transformer_optimizer.step()
        
        samples = data.shape[0]
        
        sum_loss += loss.item() * samples
        counter += samples
        
        #Print the results
        '''
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), sum_loss / counter))
        '''
            
      
    final_loss = sum_loss / counter    
    
    print('Average Loss: {:.6f}'.format(final_loss))
            
        
        


# In[182]:


def validate(model, test_loader, criterion, epoch):
    model.eval()
    print("Evaluating...")
    valid_running_correct = 0
    counter = 0
    sum_loss = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            
            outputs = model(data)
            outputs = torch.squeeze(outputs, -1)
            
            samples = data.shape[0]
            loss = criterion(outputs, target.float())
            
            sum_loss += loss.item() * samples
            counter += samples
            valid_running_correct += count_correct_incorrect(target, outputs)
            
            # Progress Update während der Validation
            '''
            if batch_idx % 100 == 0:
                current_loss = sum_loss / counter
                current_acc = 100. * valid_running_correct / counter  # Benutze counter statt dataset.length
                print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), current_loss))
                print("Current validation accuracy: {:.4f}%".format(current_acc))
            '''

    # Finale Metriken am Ende der Validation
    final_loss = sum_loss / counter
    final_accuracy = 100. * valid_running_correct / len(test_loader.dataset)
    
    print('Average Loss: {:.6f}'.format(final_loss))
    print("Final accuracy: {:.4f}%".format(final_accuracy))
    
    return final_loss, final_accuracy

def count_correct_incorrect(labels, outputs):
    outputs = torch.sigmoid(outputs)
    predictions = (outputs >= 0.5).float()
    correct = (predictions == labels).sum().item()
    return correct


# In[183]:


for epoch in range(EPOCHS):
    print(f"[INFO]: Epoch {epoch+1} of {EPOCHS}")
    train( model=model,train_loader=train_loader, criterion=criterion, epoch=epoch)
    validate( model=model, test_loader=test_loader, criterion=criterion, epoch=epoch)
    torch.save(model.state_dict(), f"models/transformer_classification_{epoch+1}.pt")
    


# In[ ]:


checkpoint_path = 'models/transformer_classification/model.pt'

