"""Keras Fashion MNIST
https://www.tensorflow.org/tutorials/keras/classification
"""

import keras

# the data, split between train and test sets
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

print(train_images.shape)
#print(train_images[0])

# view image
import matplotlib.pyplot as plt
plt.imshow(train_images[0])

#view as csv
import pandas as pd
pd.DataFrame(train_images[0]).to_csv("image.csv")

# Scale images to the [0, 1] range
train_images = train_images / 255.0
test_images = test_images / 255.0

# row 3 in csv
print(train_images[0][3])

#from keras import layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(10, activation='softmax')
])
model.summary()

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels,
                    epochs=10,
                    verbose=1,
                    validation_data=(test_images, test_labels))

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

"""https://www.tensorflow.org/tutorials/images/cnn#add_dense_layers_on_top"""
