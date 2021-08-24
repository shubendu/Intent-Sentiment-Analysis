import os
from flask import Flask, render_template
from flask import request

import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np




model = load_model('models/intents.h5')


with open('utils/tokenizer.pkl','rb') as file:
    tokenizer = pickle.load(file)

with open('utils/label_encoder.pkl','rb') as file:
    label_encoder = pickle.load(file)

class IntentClassifier:
    def __init__(self,model,tokenizer,label_encoder):
        self.classifier = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder

    def get_intent(self,text):
        self.text = [text]
        self.test_keras = self.tokenizer.texts_to_sequences(self.text)
        self.test_keras_sequence = pad_sequences(self.test_keras, maxlen=16, padding='post')
    
        self.pred = self.classifier.predict(self.test_keras_sequence)
        return self.label_encoder.inverse_transform(np.argmax(self.pred,1))[0]

app = Flask(__name__)

nlu = IntentClassifier(model,tokenizer,label_encoder) 


@app.route('/', methods=['GET', 'POST'])
def index():
  
    if request.method == 'POST':
      form = request.form 
        
      result = []
      sentence = form['sentence']
      prediction = nlu.get_intent(sentence)

      result.append(form['sentence'])
      result.append(prediction)

      return render_template("index.html",result = result)

    return render_template("index.html")

if __name__ == "__main__":
	app.run(debug=True)
