import os 
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ["OMP_NUM_THREADS"] = "1"  # Limit numpy to use only one CPU core
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.utils import to_categorical
import time

# Load and preprocess the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# Build the CNN model
model = models.Sequential()

# Convolutional layer
model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Max pooling layer
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output for the fully connected layer
model.add(layers.Flatten())

# Fully connected layer
# model.add(layers.Dense(128, activation='relu'))

# Output layer
model.add(layers.Dense(10, activation='softmax'))

optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
# Compile the model
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Measure the time taken for training
start_time = time.time()

# Train the model
model.fit(x_train, y_train, epochs=20, batch_size=16)

# Calculate the training time
training_time = time.time() - start_time
print(f"Training time: {training_time:.2f} seconds")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")
