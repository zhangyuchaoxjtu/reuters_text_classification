import tensorflow as tf
import keras as ks
from tensorflow.keras.datasets import reuters
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as pl
from keras import metrics
from keras import backend as K

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

print(len(train_data[0])) #8982
print(len(test_data)) #2246
print(train_data[1])
print(train_labels[20]) # 3

def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1
    return results

x_train = vectorize_sequences(train_data)

x_test = vectorize_sequences(test_data)

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))


'''
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
'''
#the code above is feasible too which is implemented to achieve the metric "f1-score".

def f1_score(y_true , y_pred):
    return 2*((metrics.Precision(y_true , y_pred)*metrics.Recall(y_true , y_pred))/(metrics.Precision(y_true , y_pred) + metrics.Recall(y_true , y_pred))) + k.epsilo()

# above the metrics in keras are function which is really amazing to me!

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy','Recall','Precision',f1])


'''
#cross-validation input 
x_val = x_train[:1000]  # qian 1000 hang
partial_x_train = x_train[1000:]  #chuqu qian 1000 shengxiade 

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]


history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=80,
                    batch_size=512,
                    validation_data=(x_val, y_val))
'''
#cross-validation input 

history = model.fit(x_train,
                    one_hot_train_labels,
                    epochs=80,
                    batch_size=50,
                    validation_split=0.1)


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
