"""
Application Streamlit : Prédicteur de longueur de flamme hydrogène

Cette application charge un pipeline de régression par processus gaussien
(noyau Matérn) entraîné pour prédire la longueur de flamme normalisée (Z1/d1)
avec une incertitude prédictive.

Entrées utilisateur (brutes) :
- S1, S2, u1, u2, PhiTot, PhiInt_rescaled, d1, d2, J

Variables dérivées calculées automatiquement :
- k
- phi_S1_S2
- alpha_degeneve
- S12

Sorties affichées :
- Z1/d1 avec format valeur ± incertitude (au niveau de confiance choisi)
- Intervalle de confiance équivalent sur Z1/d1
- Z1 en mm avec format valeur ± incertitude
- Intervalle de confiance équivalent sur Z1 (mm)

Fichiers requis dans le même dossier que ce script :
- gpr_pipeline_matern.joblib
- merged_df.csv

Exécution :
python -m streamlit run flame_length_pred/app.py
"""

import os
import pandas as pd
import joblib
import streamlit as st

PATH_FLAMELENGTH_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PATH_FLAMELENGTH_FOLDER, "gpr_pipeline_matern.joblib")
DATA_PATH = os.path.join(PATH_FLAMELENGTH_FOLDER, "merged_df.csv")

st.set_page_config(
    page_title="Prédicteur de longueur de flamme", page_icon="🔥", layout="wide"
)
st.title("Prédicteur de longueur de flamme hydrogène (GPR Matérn)")
st.caption(
    "Entrez uniquement les paramètres bruts. Les variables dérivées sont calculées automatiquement."
)

Z_MAP = {
    "80%": 1.2816,
    "90%": 1.6449,
    "95%": 1.9600,
}

RAW_FEATURE_ORDER = [
    "S1",
    "S2",
    "u1",
    "u2",
    "PhiTot",
    "PhiInt_rescaled",
    "d1",
    "d2",
    "J",
]

RAW_DEFAULTS = {
    "S1": 0.000,
    "S2": 0.000,
    "u1": 28.162,
    "u2": 28.475,
    "PhiTot": 0.349,
    "PhiInt_rescaled": 0.669,
    "d1": 0.010,
    "d2": 0.020,
    "J": 1.784,
}


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


@st.cache_data
def load_reference_data(csv_path: str):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


def get_feature_order(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    # Repli si les noms de colonnes ne sont pas exposés par le pipeline
    return ["u1", "u2", "phi_S1_S2", "S1", "S12", "PhiInt_rescaled"]


def build_model_features_from_raw(r: dict) -> dict:
    if r["d1"] <= 0 or r["d2"] <= 0:
        raise ValueError("d1 et d2 doivent être strictement positifs.")

    # Variables dérivées (formules du notebook)
    k = 2.0 / (1.0 + (r["d1"] / r["d2"]) ** 2)
    phi_s1_s2 = r["PhiTot"] * (
        ((1.0 + (2.0 * r["S1"]) ** 2) / (1.0 + (k * r["S2"]) ** 2)) ** 0.5
    )

    alpha_degeneve = 1.0 / (
        1.0 + r["J"] * (r["d2"] ** 2 - (r["d1"] + 0.002) ** 2) / (r["d1"] ** 2)
    )

    s12 = (
        r["d1"] / r["d2"] * alpha_degeneve * r["S1"] + (1.0 - alpha_degeneve) * r["S2"]
    )

    return {
        "S1": r["S1"],
        "S2": r["S2"],
        "u1": r["u1"],
        "u2": r["u2"],
        "PhiTot": r["PhiTot"],
        "PhiInt_rescaled": r["PhiInt_rescaled"],
        "d1": r["d1"],
        "d2": r["d2"],
        "J": r["J"],
        "k": k,
        "phi_S1_S2": phi_s1_s2,
        "alpha_degeneve": alpha_degeneve,
        "S12": s12,
    }


if not os.path.exists(MODEL_PATH):
    st.error(f"Modèle introuvable : {MODEL_PATH}")
    st.stop()

model = load_model(MODEL_PATH)
ref_df = load_reference_data(DATA_PATH)
feature_order = get_feature_order(model)

st.success(f"Modèle chargé : {MODEL_PATH}")
st.write("Ordre des variables attendu par le modèle :", feature_order)

st.subheader("Paramètres d'entrée")
col1, col2 = st.columns(2)
raw_inputs = {}

for i, f in enumerate(RAW_FEATURE_ORDER):
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        raw_inputs[f] = st.number_input(
            label=f,
            value=float(RAW_DEFAULTS[f]),
            step=0.01,
            format="%.3f",
        )

st.markdown("---")
ci_label = st.selectbox("Intervalle de confiance", list(Z_MAP.keys()), index=0)
z_value = Z_MAP[ci_label]

predict_clicked = st.button("Prédire", type="primary")

if predict_clicked:
    try:
        raw = {f: float(raw_inputs[f]) for f in RAW_FEATURE_ORDER}
        all_features = build_model_features_from_raw(raw)

        missing = [f for f in feature_order if f not in all_features]
        if missing:
            st.error(f"Variables attendues manquantes pour le modèle : {missing}")
            st.stop()

        x_row = {f: all_features[f] for f in feature_order}
        x_new = pd.DataFrame([x_row], columns=feature_order)

        scaler = model.named_steps["scaler"]
        gpr = model.named_steps["gpr"]

        x_scaled = scaler.transform(x_new)
        y_pred_arr, y_std_arr = gpr.predict(x_scaled, return_std=True)

        y_pred = float(y_pred_arr[0])  # Prédiction sur Z1/d1
        y_std = float(y_std_arr[0])  # Écart-type prédictif sur Z1/d1

        # Demi-largeur d'intervalle pour le niveau de confiance choisi
        pm = z_value * y_std  # sur Z1/d1

        # Conversion vers Z1 en mm
        d1_val = raw["d1"]  # en m
        z1_pred_mm = y_pred * d1_val * 1000.0
        pm_mm = pm * d1_val * 1000.0

        st.subheader("Résultats de prédiction")
        m1, m2 = st.columns(2)
        m1.metric("Z1/d1 prédit", f"{y_pred:.2f}")
        m2.metric(f"± ({ci_label}) sur Z1/d1", f"{pm:.2f}")

        st.write(f"Z1/d1 = {y_pred:.2f} $\\pm$ {pm:.2f}")
        st.write(
            f"IC {ci_label} sur Z1/d1 : [{(y_pred - pm):.2f}, {(y_pred + pm):.2f}]"
        )

        st.markdown("### Longueur de flamme absolue")
        st.write(f"Z1 (mm) = {z1_pred_mm:.1f} $\\pm$ {pm_mm:.1f}")
        st.write(
            f"IC {ci_label} sur Z1 (mm) : [{(z1_pred_mm - pm_mm):.1f}, {(z1_pred_mm + pm_mm):.1f}]"
        )

        with st.expander("Variables dérivées calculées"):
            st.write(
                {
                    "J": raw["J"],
                    "k": all_features["k"],
                    "phi_S1_S2": all_features["phi_S1_S2"],
                    "alpha_degeneve": all_features["alpha_degeneve"],
                    "S12": all_features["S12"],
                }
            )

        if ref_df is not None:
            st.markdown("### Vérification des plages d'entrée")
            checks = []
            for f in RAW_FEATURE_ORDER:
                if f in ref_df.columns:
                    fmin = float(ref_df[f].min())
                    fmax = float(ref_df[f].max())
                    val = float(raw[f])
                    checks.append(
                        {
                            "feature": f,
                            "value": val,
                            "train_min": fmin,
                            "train_max": fmax,
                            "in_range": (fmin <= val <= fmax),
                        }
                    )

            if checks:
                checks_df = pd.DataFrame(checks)
                st.dataframe(checks_df, use_container_width=True)
                if not checks_df["in_range"].all():
                    st.warning(
                        "Certaines entrées sont hors de la plage d'entraînement. L'incertitude peut être sous-estimée."
                    )

    except Exception as exc:
        st.error(f"Échec de la prédiction : {exc}")
