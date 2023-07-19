from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# モデルをロード
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# APIで呼び出し。JSONでExample{"面積": [20.75], "数量": [4.0]}
@app.route("/predict", methods=["POST"])
def predict():
    # JSONデータを取得
    data = request.get_json()

    # DataFrameを作成
    df = pd.DataFrame(data)

    # 予測を実行
    prediction = model.predict(df)

    # 結果を出力
    result = {"prediction": prediction.tolist()}  # NumPy配列をリストに変換

    return jsonify(result)


@app.route("/", methods=["GET", "POST"])
def input_form():
    prediction = None

    if request.method == "POST":
        # フォームから入力を取得
        data = {
            "面積": [request.form["area"]],
            "数量": [request.form["quantity"]],
        }

        # DataFrameを作成
        df = pd.DataFrame(data)

        # 予測を実行
        prediction = model.predict(df)[0]

    # HTMLフォームを表示し、予測結果があれば表示
    return render_template("input.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
