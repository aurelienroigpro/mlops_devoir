from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
# commentaire:
# Chargement du meilleur modèle: Logistic Regression
model = pickle.load(open("LogisticRegression.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupération des inputs du formulaire
        features = [
            float(request.form["credit_lines_outstanding"]),
            float(request.form["loan_amt_outstanding"]),
            float(request.form["total_debt_outstanding"]),
            float(request.form["income"]),
            float(request.form["years_employed"]),
            float(request.form["fico_score"])
        ]

        # Transformation en format compatible modèle
        final_features = np.array(features).reshape(1, -1)

        # Réalisation de la Prédiction
        prediction = model.predict(final_features)[0]

        return render_template(
            "index.html",
            prediction_text=f"Prediction : {prediction}"
        )
# Gestion des erreurs:
    except ValueError as e:
        # erreur de conversion en float
        return render_template("index.html", prediction_text=f"Erreur de saisie : {e}")
    except KeyError as e:
        # champ manquant dans le formulaire
        return render_template("index.html", prediction_text=f"Champ manquant : {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
    # app.run(debug=True)