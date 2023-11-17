import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

TRAIN_DIR = "./imgs/train/"
img_height = 480
img_width = 640
batch_size = 32
training_dataset = tf.keras.utils.image_dataset_from_directory('./imgs/train/',
                                                      validation_split=0.2,
                                                        subset="training",
                                                        seed=123,
                                                        image_size=(img_height, img_width),
                                                        batch_size=batch_size)
validation_dataset = tf.keras.utils.image_dataset_from_directory('./imgs/train/',
                                                      validation_split=0.2,
                                                        subset="validation",
                                                        seed=123,
                                                        image_size=(img_height, img_width),
                                                        batch_size=batch_size)

class_names = training_dataset.class_names
num_classes = len(class_names)

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=1
history = model.fit(
  training_dataset,
  validation_data=validation_dataset,
  epochs=epochs
)