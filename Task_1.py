import numpy as np
from keras import Sequential
from keras.layers import Dense
import pandas as pd
import csv
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def train_and_test(file):
    data = df.to_numpy()
    X = data[:,1:].astype(float)
    y = data[:,0]
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)
    one_hot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
    X_train, X_test, y_train, y_test = train_test_split(X, one_hot_encoded, test_size=0.3)
    return X_train, X_test, y_train, y_test

def neural_model(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(10,input_dim=6, activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, epochs=100, batch_size=10, shuffle=True)
    y_pred = model.predict(X_test)
    y_pred_int = np.argmax(y_pred, axis=1)
    y_test_int = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test_int, y_pred_int)
    return cm


#open("vector.csv").read()
#file = np.genfromtxt("vector.csv", delimiter=',')
#X,Y = substitute_vectors(file)

df = pd.read_csv('vector.csv', sep=',')

X_train, X_test, y_train, y_test = train_and_test(df)
result = neural_model(X_train, X_test, y_train, y_test)
print(result)


