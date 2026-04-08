from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# 🔹 Charger le modèle Logistic Regression
model = pickle.load(open("LogisticRegression.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 🔹 Récupérer les inputs du formulaire
        features = [
            float(request.form["credit_lines_outstanding"]),
            float(request.form["loan_amt_outstanding"]),
            float(request.form["total_debt_outstanding"]),
            float(request.form["income"]),
            float(request.form["years_employed"]),
            float(request.form["fico_score"])
        ]

        # 🔹 Transformer en format compatible modèle
        final_features = np.array(features).reshape(1, -1)

        # 🔹 Prédiction
        prediction = model.predict(final_features)[0]

        return render_template(
            "index.html",
            prediction_text=f"Prediction : {prediction}"
        )

    except ValueError as e:
        # erreur de conversion en float
        return render_template("index.html", prediction_text=f"Erreur de saisie : {e}")
    except KeyError as e:
        # champ manquant dans le formulaire
        return render_template("index.html", prediction_text=f"Champ manquant : {e}")

if __name__ == "__main__":
    app.run(debug=True)