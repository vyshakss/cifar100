import tensorflow as tf
from tensorflow.keras import models, layers, regularizers, callbacks,optimizers
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.datasets import cifar10
(train_images,train_labels),(test_images,test_labels)=cifar10.load_data()
train_images=train_images/255.0
test_images=test_images/255.0

x_train = train_images[:45000]
x_val   = train_images[45000:]
y_train = train_labels[:45000]
y_val   = train_labels[45000:]
train_dataset=tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset=train_dataset.shuffle(45000).batch(128).prefetch(tf.data.AUTOTUNE)
val_dataset=tf.data.Dataset.from_tensor_slices((x_val,y_val))
val_dataset=val_dataset.batch(128).prefetch(tf.data.AUTOTUNE)


def se_block(x,ratio=4):
    filters=x.shape[-1]
    se=layers.GlobalAveragePooling2D()(x)
    se=layers.Dense(max(1,filters//ratio),activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])

def mbconv(x,out_filters,expand_ratio=4,se_ratio=4):
    in_filters=x.shape[-1]
    mid_filters=in_filters*expand_ratio

    out=layers.Conv2D(mid_filters,(1,1),padding='same',use_bias=False)(x)
    out=BatchNormalization()(out)
    out=layers.Activation('relu')(out)

    out=layers.DepthwiseConv2D((3,3),padding='same',use_bias=False)(out)
    out=BatchNormalization()(out)
    out=layers.Activation('relu')(out)

    out=se_block(out,ratio=se_ratio)

    out = layers.Conv2D(out_filters,(1,1),padding='same',use_bias=False)(out)
    out = BatchNormalization()(out)

    if in_filters != out_filters:
        x = layers.Conv2D(out_filters,(1,1),padding='same',use_bias=False,kernel_regularizer=regularizers.L2(0.0005))(x)
        x = BatchNormalization()(x)
    out = layers.Add()([out, x])
    return out






inputs=layers.Input(shape=(32,32,3))
x = layers.RandomFlip('horizontal')(inputs)
x = layers.RandomRotation(0.25)(x)
x = layers.RandomContrast(0.3)(x)

x=layers.Conv2D(64,(3,3),padding='same',use_bias=False)(x)
x=BatchNormalization()(x)
x=layers.Activation('relu')(x)  

x = mbconv(x, out_filters=64,  expand_ratio=1)
x = mbconv(x, out_filters=64,  expand_ratio=4)
x = layers.MaxPooling2D(2,2)(x)
x = Dropout(0.2)(x)

x = mbconv(x, out_filters=128,  expand_ratio=4)
x = mbconv(x, out_filters=128,  expand_ratio=4)
x = mbconv(x, out_filters=128,  expand_ratio=4)
x = layers.MaxPooling2D(2,2)(x)
x = Dropout(0.3)(x)

x = mbconv(x, out_filters=256, expand_ratio=4)
x = mbconv(x, out_filters=256, expand_ratio=4)
x = mbconv(x, out_filters=256,  expand_ratio=4)
x = layers.MaxPooling2D(2,2)(x)
x = Dropout(0.3)(x)



x=layers.GlobalAveragePooling2D()(x)

x=layers.Dense(256,activation='relu',kernel_regularizer=regularizers.L2(0.001))(x)
output=layers.Dense(10,activation='softmax')(x)

model=models.Model(inputs=inputs,outputs=output)
model.summary()

trainer=callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_delta=0.01,
    min_lr=0.000000001,
    verbose=1
)
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=60,
    callbacks=[trainer]
)
model.save('cifar10mb.keras')

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"\nTest accuracy: {test_accuracy:.4f}")
print(f"Test loss:     {test_loss:.4f}")


