import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras import models, layers, regularizers, callbacks
from tensorflow.keras.layers import Dropout
from tensorflow.keras.datasets import cifar100

IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_CLASSES = 100

# ── LOAD DATA ─────────────────────────────────────────
(train_images, train_labels), (test_images, test_labels) = cifar100.load_data()

# One-hot encode — needed for CutMix/MixUp + label smoothing
train_labels_oh = tf.keras.utils.to_categorical(train_labels, NUM_CLASSES).astype('float32')
test_labels_oh  = tf.keras.utils.to_categorical(test_labels,  NUM_CLASSES).astype('float32')

x_train = train_images[:45000]
x_val   = train_images[45000:]
y_train = train_labels_oh[:45000]
y_val   = train_labels_oh[45000:]
y_val_sparse   = train_labels[45000:]   # keep sparse for val (no mixing needed)
test_labels_oh = tf.keras.utils.to_categorical(test_labels, NUM_CLASSES).astype('float32')


# ── PREPROCESSING ─────────────────────────────────────
def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.15)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    image = tf.image.random_saturation(image, 0.85, 1.15)
    return image, label


# ── CUTMIX ────────────────────────────────────────────
def cutmix(images, labels, alpha=1.0):
    batch_size = tf.shape(images)[0]
    lam        = tf.cast(tf.random.gamma([1], alpha, 1.0)[0], tf.float32)
    indices    = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)

    img_h = tf.shape(images)[1]
    img_w = tf.shape(images)[2]
    cut_ratio = tf.math.sqrt(1.0 - lam)
    cut_h = tf.cast(tf.cast(img_h, tf.float32) * cut_ratio, tf.int32)
    cut_w = tf.cast(tf.cast(img_w, tf.float32) * cut_ratio, tf.int32)
    cx = tf.random.uniform([], 0, img_w, dtype=tf.int32)
    cy = tf.random.uniform([], 0, img_h, dtype=tf.int32)
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, img_w)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, img_h)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, img_w)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, img_h)

    rows     = tf.range(img_h)
    cols     = tf.range(img_w)
    row_mask = tf.logical_and(rows >= y1, rows < y2)
    col_mask = tf.logical_and(cols >= x1, cols < x2)
    box_mask = tf.cast(row_mask[:, None] & col_mask[None, :], tf.float32)[:, :, None]

    lam = 1.0 - tf.cast((x2-x1)*(y2-y1), tf.float32) / tf.cast(img_h*img_w, tf.float32)
    new_images = images * (1.0 - box_mask) + shuffled_images * box_mask
    new_labels = labels * lam + shuffled_labels * (1.0 - lam)
    return new_images, new_labels


# ── MIXUP ─────────────────────────────────────────────
def mixup(images, labels, alpha=0.2):
    lam     = tf.cast(tf.random.gamma([1], alpha, 1.0)[0], tf.float32)
    lam     = tf.minimum(lam, 1.0 - lam)
    indices = tf.random.shuffle(tf.range(tf.shape(images)[0]))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)
    return (images*(1-lam) + shuffled_images*lam,
            labels*(1-lam) + shuffled_labels*lam)


def cutmix_or_mixup(images, labels):
    use_aug    = tf.random.uniform([]) < 0.5
    use_cutmix = tf.random.uniform([]) < 0.5
    return tf.cond(use_aug,
        lambda: tf.cond(use_cutmix,
            lambda: cutmix(images, labels),
            lambda: mixup(images, labels)),
        lambda: (images, labels))


# ── DATASETS ──────────────────────────────────────────
with tf.device('/CPU:0'):
    train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .shuffle(10000)
                .map(preprocess,       num_parallel_calls=tf.data.AUTOTUNE)
                .map(augment,          num_parallel_calls=tf.data.AUTOTUNE)
                .batch(BATCH_SIZE)
                .map(cutmix_or_mixup,  num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))

    val_ds = (tf.data.Dataset.from_tensor_slices((x_val, y_val))
              .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(BATCH_SIZE)
              .prefetch(tf.data.AUTOTUNE))

    test_ds = (tf.data.Dataset.from_tensor_slices((test_images, test_labels_oh))
               .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
               .batch(BATCH_SIZE)
               .prefetch(tf.data.AUTOTUNE))


# ── MODEL ─────────────────────────────────────────────
base_model = tf.keras.applications.ConvNeXtTiny(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_preprocessing=True
)
base_model.trainable = False

inputs  = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(512, activation='gelu',
              kernel_regularizer=regularizers.L2(0.0001))(x)
x       = layers.LayerNormalization()(x)
x       = Dropout(0.4)(x)
x       = layers.Dense(256, activation='gelu',
              kernel_regularizer=regularizers.L2(0.0001))(x)
x       = layers.LayerNormalization()(x)
x       = Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

model   = models.Model(inputs=inputs, outputs=outputs)
model.summary()

steps_per_epoch = len(x_train) // BATCH_SIZE


# ── PHASE 1 — head only ───────────────────────────────
PHASE1_EPOCHS = 50

lr_1 = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=steps_per_epoch * PHASE1_EPOCHS,
    alpha=1e-6 / 1e-3,
    warmup_steps=steps_per_epoch * 3,
    warmup_target=1e-3
)
model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=lr_1, weight_decay=0.01
    ),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

early_stop_1 = callbacks.EarlyStopping(
    monitor='val_accuracy', patience=12,
    restore_best_weights=True, verbose=1
)

print("\nPhase 1 — head only...")
history_1 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=PHASE1_EPOCHS, callbacks=[early_stop_1]
)
print(f"Phase 1 best: {max(history_1.history['val_accuracy']):.4f}")


# ── PHASE 2 — unfreeze top 100 layers ─────────────────
PHASE2_EPOCHS = 70
base_model.trainable = True

# Unfreeze top 100 layers, keep LayerNorms frozen
freeze_until = len(base_model.layers) - 100
for i, layer in enumerate(base_model.layers):
    layer.trainable = (i >= freeze_until)
    if isinstance(layer, layers.LayerNormalization):
        layer.trainable = False   # keep LayerNorms stable

trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"\nPhase 2 trainable: {trainable:,}")

lr_2 = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=5e-6,
    decay_steps=steps_per_epoch * PHASE2_EPOCHS,
    alpha=1e-8 / 5e-6,
    warmup_steps=steps_per_epoch * 2,
    warmup_target=5e-6
)
model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=lr_2, weight_decay=0.01, clipnorm=1.0
    ),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

early_stop_2 = callbacks.EarlyStopping(
    monitor='val_accuracy', patience=15,
    restore_best_weights=True, verbose=1
)
checkpoint = callbacks.ModelCheckpoint(
    'cifar100_convnext.keras',
    monitor='val_accuracy', save_best_only=True, verbose=1
)

print("\nPhase 2 — fine-tuning top 100 layers...")
history_2 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=PHASE2_EPOCHS,
    callbacks=[early_stop_2, checkpoint]
)


# ── PHASE 3 — full model ──────────────────────────────
PHASE3_EPOCHS = 30
base_model.trainable = True   # unfreeze everything

lr_3 = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-6,
    decay_steps=steps_per_epoch * PHASE3_EPOCHS,
    alpha=1e-8 / 1e-6,
)
model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=lr_3, weight_decay=0.005, clipnorm=1.0
    ),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

early_stop_3 = callbacks.EarlyStopping(
    monitor='val_accuracy', patience=10,
    restore_best_weights=True, verbose=1
)

print("\nPhase 3 — full model fine-tuning...")
history_3 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=PHASE3_EPOCHS,
    callbacks=[early_stop_3, checkpoint]
)

model.save('cifar100_convnext_final.keras')

test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"Test loss:     {test_loss:.4f}")