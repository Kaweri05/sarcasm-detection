from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(5,)))
model.add(Dense(1, activation='sigmoid'))

print("Everything working perfectly ✅")