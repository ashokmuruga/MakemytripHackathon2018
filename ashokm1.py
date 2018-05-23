# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import csv

# fix random seed for reproducibility
np.random.seed(7)

# load pima indians dataset
dataset = np.loadtxt("test2.csv")

# split into input (X) and output (Y) variables
X = dataset[:,1:6]
Y = dataset[:,6]

# create model
model = Sequential()
model.add(Dense(12, input_dim=5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=10000, batch_size=10, verbose=2)

# calculate predictions
predictions = model.predict(X)

# round predictions
rounded = [x[0] for x in predictions]
print(dataset[:,0].tolist(),rounded)
zip(dataset[:,0].tolist(),rounded)
with open('final.csv','w') as f:
	writer=csv.writer(f,delimiter='\t')
	writer.writerows(zip(dataset[:,0].tolist(),rounded))	
