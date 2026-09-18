import tensorflow as tf
from tensorflow.keras import layers, models

def build_stateless_mlp(input_shape=(10, 7), num_classes: int = 4) -> models.Model:
    """Stateless Multi-Layer Perceptron baseline across flattened observation window."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ], name="Baseline_MLP")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model