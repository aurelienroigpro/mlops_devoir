import pickle
import numpy as np

# 🔹 Charger le modèle Logistic Regression
model = pickle.load(open("LogisticRegression.pkl", "rb"))

def test_predict():
    # 🔹 Exemple de nouvelles données
    new_data = [
        0,          # credit_lines_outstanding
        5221.54,    # loan_amt_outstanding
        3915.47,    # total_debt_outstanding
        78039.38,   # income
        5,          # years_employed
        605         # fico_score
    ]

    # 🔹 Transformer en format compatible modèle
    final_features = np.array(new_data).reshape(1, -1)

    # 🔹 Prédiction
    prediction = model.predict(final_features)[0]

    # 🔹 Test : prédiction attendue (ici 0 ou 1 selon ton dataset)
    assert prediction in [0, 1], f"Prediction inattendue : {prediction}"

if __name__ == "__main__":
    test_predict()
    print("Test passé ✅")