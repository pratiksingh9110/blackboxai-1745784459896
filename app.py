from flask import Flask, render_template, request, jsonify
import threading
import main

app = Flask(__name__)

training_history = None
training_in_progress = False

@app.route('/')
def index():
    data_preview = main.get_data_preview()
    return render_template('index.html', data_preview=data_preview, training_in_progress=training_in_progress, training_history=training_history)

def train_model_thread():
    global training_history, training_in_progress
    training_in_progress = True
    training_history = main.train_model()
    training_in_progress = False

@app.route('/train', methods=['POST'])
def train():
    global training_in_progress
    if training_in_progress:
        return jsonify({'status': 'Training already in progress'}), 400
    thread = threading.Thread(target=train_model_thread)
    thread.start()
    return jsonify({'status': 'Training started'})

if __name__ == '__main__':
    app.run(debug=True)
