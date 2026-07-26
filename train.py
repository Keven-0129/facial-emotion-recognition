import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)
# IMAGE PREPROCESSING

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_data = train_datagen.flow_from_directory(
    'train',
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    'test',
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)

# CNN MODEL

model = Sequential()

# First Convolution Block
model.add(Conv2D(
    32,
    (3,3),
    activation='swish',
    input_shape=(48,48,1)
))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

# Second Block
model.add(Conv2D(
    64,
    (3,3),
    activation='swish'
))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

# Third Block
model.add(Conv2D(
    128,
    (3,3),
    activation='swish'
))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

#Fourth Block
model.add(Conv2D(
    256,
    (3,3),
    activation='swish'
))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten
model.add(Flatten())

# Dense Layer
model.add(Dense(256, activation='swish'))
model.add(Dropout(0.5))

model.add(Dense(128, activation='swish'))
model.add(Dropout(0.3))

# Output Layer
model.add(Dense(7, activation='softmax'))

# COMPILE MODEL

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# SAVE BEST MODEL

checkpoint = ModelCheckpoint(
    'best_emotion_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# TRAIN MODEL

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=32,
    callbacks=[checkpoint,early_stop]
)

# SAVE FINAL MODEL

model.save("emotion_model.h5")

print("Training Complete!")