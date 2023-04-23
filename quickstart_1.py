import time

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow_hub as hub

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(x_train[0].shape)
print(y_train.shape)

x_train, x_test = x_train / 255.0, x_test / 255.0

print(y_train[:5])
images = [x_train[i] for i in range(5)]

# plotImages(images)


# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")
print(x_train.shape)

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

'''
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, 3, input_shape=(28, 28, 1), activation='relu'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])

model.summary()

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

epochs = 5
start_time = time.time()
history = model.fit(train_ds, epochs=epochs, validation_data=test_ds)
end_time = time.time()
print(round(end_time-start_time), 's to train')

export_path_keras = "./{}.h5".format(int(end_time))
print(export_path_keras)
model.save(export_path_keras)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
'''

export_path_keras = './1682154035.h5'
model = tf.keras.models.load_model(
  export_path_keras,
  # `custom_objects` tells keras how to load a `hub.KerasLayer`
  custom_objects={'KerasLayer': hub.KerasLayer})

model.summary()


image_batch, label_batch = next(test_ds.as_numpy_iterator())
print(image_batch.shape)
print(label_batch.shape)
print(label_batch)

predicted_batch = model.predict(image_batch)
predicted_ids = np.argmax(predicted_batch, axis=-1)
print(predicted_ids)

