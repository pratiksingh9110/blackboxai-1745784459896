import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def load_data_from_db(db_path, query):
    """
    Connect to the SQLite database and load data using the provided SQL query.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def preprocess_data(df):
    """
    Preprocess the data for training.
    This is a placeholder function and should be customized based on the dataset.
    """
    # Example: Assume last column is the target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def build_model(input_dim):
    """
    Build a simple deep learning model using TensorFlow Keras.
    """
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # For binary classification; adjust as needed
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

DB_PATH = 'dataset.db'  # Change as needed
QUERY = 'SELECT * FROM data_table;'  # Change table name as needed

def get_data_preview():
    """
    Load data from the database and return a preview (first 5 rows) as HTML.
    """
    df = load_data_from_db(DB_PATH, QUERY)
    return df.head().to_html(classes='table table-striped')

def train_model(epochs=10, batch_size=32):
    """
    Train the model and return training history as a dictionary.
    """
    df = load_data_from_db(DB_PATH, QUERY)
    X, y = preprocess_data(df)
    model = build_model(X.shape[1])
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    model.save('trained_model.h5')
    return history.history

if __name__ == '__main__':
    print("Run this script via the web interface or call train_model() directly.")
