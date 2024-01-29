
import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer
intents = json.loads(open("intents.json").read())

palabras = []
clases = []
documentos = []
ignore_latters = ["?", "!", ".",","]

for intent in intents["intents"]:
  for pattern in intent["patterns"]:
    word_list = nltk.tokenize(pattern)
    palabras.append(word_list)
    documentos.append((word_list), intent["tag"])
    if intent["tag"] not in clases:
      clases.append(intent["tag"])

palabras = [lemmatizer.lemmatizer(palabra)
  for palabra in palabras
    if palabra not in ignore_latters]
palabras = sorted(set(palabras))

clases = sorted(set(clases))

pickle.dump(palabras, open("words.pkl", "wb"))
pickle.dump(palabras, open("clases.pkl", "wb"))

training = []
output_empty = [0] * len(clases)

for documento in documentos:
  bag = []
  word_patterns = documento[0]
  word_petterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
  for palabra in palabras:
    bag.append(1)if palabra in word_patterns else bag.append(0)

  output_row = list(output_empty)
  output_row[clases.index(documento[1])] = 1
  training.append([bag, output_row])

random.shuffle(training)
training = np.array(training )

train_x = list(training[:,0])
train_y = list(training[:,0])

model = Sequential()
model.add(Dense(128, input_sahpe=(len(train_x[0]),), activation="relu"))
model.add(Dense(0.5))
model.add(Dense(64, activation="softmax"))

sgd = SGD(lr=0.01, decay=1e, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

hist = model.fit(np.array(train_x), np.array(train_y), epoch=200, batch_size = 5, verbose=1)
model.save("chatbot_model.h5", hist)
print("Done")

print(documentos)
