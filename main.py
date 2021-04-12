import pickle
from flask import Flask, request, jsonify
from model_files.ml_model import predict_mpg

# create a app 
app = Flask("mpgprediction") 

@app.route("/", methods=['POST'])
def predict():
    config = request.get_json()

    # load the model
    with open("./model_files/model.bin", "rb") as f_in:
        model = pickle.load(f_in)
        f_in.close()
    
    predictions = predict_mpg(config, model)

    results = {
        "mpg_predictions" : list(predictions)
    }

    return jsonify(results)

@app.route('/ping', methods=['GET'])
def ping():
    return "Pinging Model!!"

#To start the app in local server,
#we need the boiler plate code
if __name__ == '__main__':

    app.run(debug=True, host="localhost", port= 9696)
