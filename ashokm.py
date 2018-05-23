# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# fix random seed for reproducibility
np.random.seed(7)

# load pima indians dataset
dataset = np.loadtxt("train1.csv")

# split into input (X) and output (Y) variables
X = dataset[:,0:6]
Y = dataset[:,6]

# create model
model = Sequential()
model.add(Dense(12, input_dim=6, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=6000, batch_size=20)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
