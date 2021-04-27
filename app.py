import numpy as np
from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import LabelEncoder
app = Flask(__name__)

model = pickle.load(open("weight_category_prediction.pickle", "rb"))


#default page for webpage

@app.route("/")
def home_page():
    return render_template("index.html")

#predict
@app.route("/predict", methods = ["POST"])
def predict():
    features = [val for val in request.form.values()]
    lb = LabelEncoder()
    lb.fit(["Male", "Female"])
    int_val = lb.transform([features[2]])[0]
    print("Gender Value ",int_val)
    features[2] = int_val
    final_list = [np.array(features)]
    output = model.predict(final_list[::-1])[0]
    return render_template("index.html", prediction_text="Weight Category is : {}".format(output))
if __name__ == "__main__":
    app.run(debug=True)

