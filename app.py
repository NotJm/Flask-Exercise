from flask import Flask, request, jsonify
from joblib import load

app = Flask(__name__)
modelo = load('./modelo_iris.pkl')

@app.route('/')
def home():
  return 'Hello World'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    predict = modelo.predict(data['features'])
    return jsonify({ 'result': int(predict[0])})



if __name__ == '__main__':
  app.run(debug=True)
  
  

