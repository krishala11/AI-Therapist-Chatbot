import json
import os
import pickle
import random
import spacy
import numpy as np
import torch
import torch.nn as nn


nlp = spacy.load("en_core_web_sm")
#neural network 

class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def preprocess_text(sentence):
    doc = nlp(sentence)
    return [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]

def generate_bow_vector(sentence, words):
    sentence_words = preprocess_text(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)

def classify_intent(sentence, model, words, classes):
    bow = generate_bow_vector(sentence, words)
    bow_tensor = torch.from_numpy(bow).float().unsqueeze(0)
    output = model(bow_tensor)
    _, predicted = torch.max(output, 1)
    return classes[predicted.item()]

def fetch_response(intent_tag, intents):
    for intent in intents["intents"]:
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"])
    return "I'm sorry, I don't understand."

def execute_training_loop(epochs, progress_bar=None, status_placeholder=None):
    with open("intents.json") as file:
        intents = json.load(file)

    for file in ["words.pkl", "classes.pkl", "best_chatbot.pth"]:
        if os.path.exists(file):
            os.remove(file)

    words, classes, documents = [], [], []

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            w = preprocess_text(pattern)
            words.extend(w)
            documents.append((w, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

    words = sorted(set(words))
    classes = sorted(set(classes))

    training = []
    for doc in documents:
        bag = [0] * len(words)
        for w in doc[0]:
            if w in words:
                bag[words.index(w)] = 1
        label = [0] * len(classes)
        label[classes.index(doc[1])] = 1
        training.append([bag, label])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x = np.array(list(training[:, 0]))
    train_y = np.array(list(training[:, 1]))

    model = ChatbotModel(len(train_x[0]), 512, len(train_y[0]))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        inputs = torch.from_numpy(train_x).float()
        targets = torch.from_numpy(np.argmax(train_y, axis=1))
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if progress_bar:
            progress_bar.progress((epoch + 1) / epochs)

    torch.save(model.state_dict(), "best_chatbot.pth")
    with open("words.pkl", "wb") as f:
        pickle.dump(words, f)
    with open("classes.pkl", "wb") as f:
        pickle.dump(classes, f)
