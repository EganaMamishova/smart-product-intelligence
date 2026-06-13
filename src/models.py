import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers

def build_mlp_model(input_dim, num_classes=3):
    """
    Milestone 1: Tabular MLP Model architecture.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_scratch_cnn(img_size=128, num_classes=3):
    """
    Milestone 2: Small CNN model built from scratch.
    """
    model = keras.Sequential([
        layers.Input(shape=(img_size, img_size, 3)),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_transfer_model(img_size=128, num_classes=3):
    """
    Milestone 2: Transfer Learning model using pretrained MobileNetV2 backbone.
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    # Freeze the pretrained backbone
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_embeddings_model(vocab_size=10000, embedding_dim=64, max_len=150, vectorize_layer=None):
    """
    Milestone 3: Keras Learned Embeddings model for sentiment classification.
    """
    inputs = layers.Input(shape=(1,), dtype=tf.string)
    
    if vectorize_layer is not None:
        x = vectorize_layer(inputs)
    else:
        # Fallback if vectorize_layer is applied outside
        x = inputs
        
    x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
