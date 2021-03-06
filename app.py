import os
from flask import Flask, render_template
from flask import request

import pickle
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from textblob import TextBlob

global graph

model = load_model('models/intents.h5')
graph = tf.get_default_graph()

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
        acc=0
        sentiment = ''
        pol = 0
        self.text = [text]
        self.test_keras = self.tokenizer.texts_to_sequences(self.text)
        self.test_keras_sequence = pad_sequences(self.test_keras, maxlen=16, padding='post')
        with graph.as_default():
            self.pred = self.classifier.predict(self.test_keras_sequence)
        acc = np.max(self.pred)#rm
        score = TextBlob(text).sentiment[0]
        if score < 0:
            sentiment = 'Negative'
            pol = score
        elif score == 0:
            sentiment = 'Neutral'
            pol = score
        else:
            sentiment = 'Positive'
            pol = score
        return self.label_encoder.inverse_transform(np.argmax(self.pred,1))[0],acc,sentiment,pol

app = Flask(__name__)

nlu = IntentClassifier(model,tokenizer,label_encoder) 


@app.route('/', methods=['GET', 'POST'])
def index():
  
    if request.method == 'POST':
      form = request.form 
        
      result = []
      sentence = form['sentence']
      prediction = nlu.get_intent(sentence)[0]
      per = np.round(nlu.get_intent(sentence)[1]*100,2)
      sentiment = nlu.get_intent(sentence)[2]
      pol = np.round(nlu.get_intent(sentence)[3],3)

      result.append(form['sentence'])
      result.append(prediction)
      result.append(per)
      result.append(sentiment)
      result.append(pol)

      return render_template("index.html",result = result)

    return render_template("index.html")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
