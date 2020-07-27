from keras.datasets import mnist
(train_images, train_lables), (test_images, test_lables) = mnist.load_data()

train_images = train_images.reshape((60_000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10_000, 28*28))
test_images = test_images.astype('float32') / 255

from keras.utils import to_categorical

train_labels = to_categorical(train_lables)
test_labels = to_categorical(test_lables)
train_labels

from keras import models
from keras import layers

# (1, 1) ~ (1, )
# print(help(layers.Dense))


network = models.Sequential()
network.add(layers.Dense(64, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(16, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy'
                , metrics=['accuracy'])

network.fit(train_images, train_labels, epochs=7, batch_size=128)