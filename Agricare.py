import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load training and validation datasets
training_set = tf.keras.utils.image_dataset_from_directory(
    'train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)
validation_set = tf.keras.utils.image_dataset_from_directory(
    'valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

# Define the CNN model
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.25))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=1500, activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.4))
cnn.add(tf.keras.layers.Dense(units=4, activation='softmax'))

# Compile the model
cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
            loss='categorical_crossentropy', 
            metrics=['accuracy'])

# Print the model summary
cnn.summary()

# Train the model
training_history = cnn.fit(x=training_set, validation_data=validation_set, epochs=10)

# Evaluate the model
val_loss, val_acc = cnn.evaluate(validation_set)
print('Accuracy:', val_acc)

# Save the model
cnn.save('trained_plant_disease_model.keras')

# Add motor control logic based on the predicted class
def activate_motors(predicted_class):
    if predicted_class == 0:
        print("Motor 1 is active")
        # Code to activate motor 1
    elif predicted_class == 1:
        print("Motor 2 is active")
        # Code to activate motor 2
    elif predicted_class == 2:
        print("Both motors are active")
        # Code to activate both motors
    elif predicted_class == 3:
        print("No motors are active")
        # Code to deactivate both motors

# Example: Using the trained model to make predictions
for images, labels in validation_set.take(1):  # Taking a single batch from the validation set
    predictions = cnn.predict(images)
    predicted_classes = tf.argmax(predictions, axis=1)
    for predicted_class in predicted_classes:
        activate_motors(predicted_class)