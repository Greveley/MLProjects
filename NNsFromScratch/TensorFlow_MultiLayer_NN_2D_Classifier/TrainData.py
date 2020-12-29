from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

from generate_points import generate_data

# Extracting training and testing data from other script
X_train, y_train, X_test, y_test = generate_data(num_points = 20000, train_perc=20, plot=False)

print ("hello")

model = Sequential()

print ("hello2")

model.add(Dense(2, activation='relu', use_bias=True))
model.add(Dense(5, activation='relu', use_bias=True))
model.add(Dense(3, activation='relu', use_bias=True))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy')

model.fit(x=X_train,y=y_train,epochs=100,verbose=0)


#plt.plot(np.array(model.history.history["loss"]))
#plt.show()

model.predict(X_test)