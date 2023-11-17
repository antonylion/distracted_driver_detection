#import numpy as np
#import matplotlib.pyplot as plt
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
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

IMG_SIZE = (480, 640)
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False, # <== Important!!!!
                                                   weights='imagenet')
base_model.trainable = False
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
base_learning_rate = 0.001

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

initial_epochs = 1
model.fit(training_dataset, validation_data=validation_dataset, epochs=initial_epochs)

base_model = model.layers[3]
base_model.trainable = True

fine_tune_at = 120

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Define a SparseCategoricalCrossentropy loss function. Use from_logits=True
loss_function=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.1 * base_learning_rate)
metrics=['accuracy']

model.compile(loss=loss_function,
              optimizer = optimizer,
              metrics=metrics)


fine_tune_epochs = 1
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(training_dataset,
                         epochs=total_epochs,
                         validation_data=validation_dataset)


