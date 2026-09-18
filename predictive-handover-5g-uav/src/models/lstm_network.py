import tensorflow as tf
from tensorflow.keras import layers, models

def build_stacked_lstm(input_shape=(10, 7), num_classes: int = 4) -> models.Model:
    """Stacked 2-Layer LSTM with Dropout for non-linear temporal link forecasting."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ], name="Proposed_LSTM")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model