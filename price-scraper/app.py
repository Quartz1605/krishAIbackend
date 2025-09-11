from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "ENAM API is running 🚜"})

@app.route("/data", methods=["GET"])
def get_data():
    df = pd.read_csv("enam_trade_data.csv")
    return jsonify(df.to_dict(orient="records"))

@app.route("/data/<state>", methods=["GET"])
def get_data_by_state(state):
    df = pd.read_csv("enam_trade_data.csv")
    filtered = df[df["State"].str.contains(state, case=False, na=False)]
    return jsonify(filtered.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)
