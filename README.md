# Deep Learning Model Training with SQL Dataset

## Overview
This project demonstrates how to connect to an SQL database (SQLite), load a dataset, and train a basic deep learning model using TensorFlow.

## Setup

1. Create and populate your SQLite database (`dataset.db`) with a table named `data_table` containing your dataset.

2. Install the required Python packages:
```
pip install -r requirements.txt
```

## Usage

Run the main script to load data from the database, train the model, and save the trained model:
```
python main.py
```

## Notes

- Modify the database path and SQL query in `main.py` as needed.
- Customize the `preprocess_data` function in `main.py` to fit your dataset and problem.
- The current model is a simple feedforward neural network for binary classification. Adjust the architecture and loss function as needed.

## License

This project is provided as-is without any warranty.
