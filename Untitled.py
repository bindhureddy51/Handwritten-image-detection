
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten
(X_train,Y_train),(X_test,Y_test)=keras.datasets.mnist.load_data()
import matplotlib.pyplot as plt
plt.imshow(X_test[1])
#Normalisation purpose
X_train=X_train/256
X_test=X_test/256

model=Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()
model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
history=model.fit(X_train,Y_train,epochs=15,validation_split=0.2)
y_prob=model.predict(X_test)
y_prob[1]
y_pred=y_prob.argmax(axis=1)
y_pred[1]
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,y_pred)



