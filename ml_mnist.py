import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import initializers

from matplotlib import pyplot as plt
import numpy as np

import tensorflow.keras.backend as K


# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the input data
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

# Convert the labels to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


def custom_activation(x):
    return K.square(x)


# Define the model architecture
model = Sequential()

initializer = initializers.Constant(
    value=np.random.randint(-2**6, 2**6, size=(784, 512)))
model.add(Dense(512, activation=custom_activation,
                input_shape=(784,), kernel_initializer=initializer))
initializer = initializers.Constant(
    value=np.random.randint(-2**6, 2**6, size=(512, 512)))
model.add(Dense(512, activation=custom_activation, kernel_initializer=initializer))
initializer = initializers.Constant(
    value=np.random.randint(-2**6, 2**6, size=(512, 10)))
model.add(Dense(10, activation='softmax', kernel_initializer=initializer))


# Compile the model with the new loss function


def integer_penalty_categorical_crossentropy(y_true, y_pred, integer_penalty=1.0):
    # Compute the categorical crossentropy loss
    cc_loss = categorical_crossentropy(y_true, y_pred)

    # Compute the integer penalty loss
    w = tf.concat([tf.reshape(w, [-1])
                  for w in model.trainable_weights], axis=0)
    ip_loss = integer_penalty * tf.reduce_sum(tf.square(w - tf.round(w)))

    # Return the total loss
    return cc_loss + ip_loss


# Train the model
for i in range(10):
    # Set the integer_penalty hyperparameter to 10 to the -i power
    integer_penalty = 2 ** (i - 2)

    # Compile the model with the new integer_penalty value
    model.compile(loss=lambda y_true, y_pred: integer_penalty_categorical_crossentropy(y_true, y_pred, integer_penalty),
                  optimizer='adam',
                  metrics=['accuracy'])

    # Train the model for one epoch
    history = model.fit(x_train, y_train, epochs=2**(5-(i // 2)),
                        batch_size=128, validation_data=(x_test, y_test))

# Plot the model performance
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Performance')
plt.xlabel('Epoch')
plt.show()

# Visualize the weights of the first dense layer as an image
weights = model.layers[0].get_weights()[0]
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(weights.T, cmap='gray')
plt.show()


# Create a new model using the previous model's weights rounded to integers
rounded_weights = [tf.round(w) for w in model.get_weights()]
model_rounded = Sequential()
model_rounded.add(Dense(512, activation=custom_activation,
                  input_shape=(784,), weights=rounded_weights[:2]))
model_rounded.add(Dense(512, activation=custom_activation,
                  weights=rounded_weights[2:4]))
model_rounded.add(Dense(10, activation='softmax', weights=rounded_weights[4:]))

# Compile the new model with the same loss function and optimizer used to train the original model
model_rounded.compile(loss=integer_penalty_categorical_crossentropy,
                      optimizer='adam', metrics=['accuracy'])

# Evaluate the new model on the test set
loss, accuracy = model_rounded.evaluate(x_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)


# Visualize the weights of the first dense layer as an image
weights = model_rounded.layers[0].get_weights()[0]
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(weights.T, cmap='gray')
plt.show()
