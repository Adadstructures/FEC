import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.optimize import differential_evolution
import shap
import matplotlib.pyplot as plt
import os

# Load the saved model and scaler
with open('bond_strength_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('bs_scaler_X.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Feature configuration
display_feature_names = ['Steel Shape', 'Cylinder Strength', 'Steel Strength', 'Rebar Strength',
                        'Concrete Area', 'Steel Area', 'Rebar Area']
# Internal feature names for model compatibility
model_feature_names = ['Steel_Shape', 'Cylinder_Strength', 'Steel_Strength', 'Rebar_Strength',
                      'Concrete_Area', 'Steel_Area', 'Rebar_Area']

shape_mapping = {'I': 1, 'H': 2, 'C': 3, 'C (+)': 4}
shape_options = list(shape_mapping.keys())

# Default LCI and cost values
default_lci = {'Steel': 2.5, 'Concrete': 0.12, 'Rebar': 1.8}
default_cost = {'Steel': 0.75, 'Concrete': 0.06, 'Rebar': 0.5}

st.title("CES Column Predictor & Optimiser")

# Section Selection
section = st.sidebar.selectbox("Choose Section", ["Prediction", "Optimisation"])

# SHAP plot function
def plot_feature_importance(model, X, feature_names):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': np.abs(shap_values.values[0])
    }).sort_values(by='SHAP Value', ascending=True)

    fig, ax = plt.subplots()
    ax.barh(shap_df['Feature'], shap_df['SHAP Value'])
    ax.set_xlabel("SHAP Value (Impact)")
    ax.set_title("Feature Importance")
    st.pyplot(fig)

# Prediction Section
if section == "Prediction":
    st.subheader("Make a Prediction")

    # Inputs with updated min/max values and superscript mm²
    shape = st.selectbox("Steel Shape", shape_options)
    input_data = {
        'Steel_Shape': shape_mapping[shape],
        'Cylinder_Strength': st.number_input("Cylinder Strength (MPa)", 
                                            min_value=21.16624309, max_value=131.136, 
                                            value=(21.16624309 + 131.136) / 2),
        'Steel_Strength': st.number_input("Steel Strength (MPa)", 
                                         min_value=253.0, max_value=803.0, 
                                         value=(253.0 + 803.0) / 2),
        'Rebar_Strength': st.number_input("Rebar Strength (MPa)", 
                                         min_value=312.0, max_value=578.0, 
                                         value=(312.0 + 578.0) / 2),
        'Concrete_Area': st.number_input(f"Concrete Area (mm\u00B2)", 
                                        min_value=21465.84073, max_value=341825.7617, 
                                        value=(21465.84073 + 341825.7617) / 2),
        'Steel_Area': st.number_input(f"Steel Area (mm\u00B2)", 
                                     min_value=520.0, max_value=10380.0, 
                                     value=(520.0 + 10380.0) / 2),
        'Rebar_Area': st.number_input(f"Rebar Area (mm\u00B2)", 
                                     min_value=314.1592654, max_value=7926.238265, 
                                     value=(314.1592654 + 7926.238265) / 2)
    }

    if st.button("Predict Load Capacity"):
        # Create DataFrame with model-compatible column names
        X = pd.DataFrame([input_data], columns=model_feature_names)
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]

        st.success(f"Predicted Load Capacity: {prediction:.2f} kN")

        # Uncertainty estimation (±5% prediction interval)
        lower = prediction * 0.95
        upper = prediction * 1.05
        st.write(f"95% Prediction Interval: ({lower:.2f} kN, {upper:.2f} kN)")

        # Feature Importance
        plot_feature_importance(model, X_scaled, display_feature_names)

# Optimisation Section
elif section == "Optimisation":
    st.subheader("Multi-objective Optimisation")

    # Updated target load range
    target_load = st.number_input("Target Load (kN)", 
                                 min_value=1136.0, max_value=18188.0, 
                                 value=(1136.0 + 18188.0) / 2)
    max_cost = st.number_input("Maximum Cost (USD)", 0.0, 50000.0, 3000.0)
    max_carbon = st.number_input("Maximum Carbon (kg CO₂)", 0.0, 50000.0, 5000.0)

    use_custom = st.checkbox("Use Custom LCI/Cost")
    if use_custom:
        lci = {
            'Steel': st.number_input("Steel LCI (kg CO₂/kg)", value=2.5),
            'Concrete': st.number_input("Concrete LCI (kg CO₂/kg)", value=0.12),
            'Rebar': st.number_input("Rebar LCI (kg CO₂/kg)", value=1.8)
        }
        cost = {
            'Steel': st.number_input("Steel Cost (USD/kg)", value=0.75),
            'Concrete': st.number_input("Concrete Cost (USD/kg)", value=0.06),
            'Rebar': st.number_input("Rebar Cost (USD/kg)", value=0.5)
        }
    else:
        lci, cost = default_lci, default_cost

    # Updated bounds for optimization
    bounds = [(1, 4),  # Steel Shape
              (21.16624309, 131.136),  # Cylinder Strength
              (253, 803),  # Steel Strength
              (312, 578),  # Rebar Strength
              (21465.84073, 341825.7617),  # Concrete Area
              (520, 10380),  # Steel Area
              (314.1592654, 7926.238265)]  # Rebar Area

    def objective(x):
        # Create DataFrame with model-compatible column names
        df = pd.DataFrame([x], columns=model_feature_names)
        x_scaled = scaler.transform(df)
        y_pred = model.predict(x_scaled)[0]

        steel_mass, concrete_mass, rebar_mass = x[5], x[4], x[6]
        total_cost = steel_mass * cost['Steel'] + concrete_mass * cost['Concrete'] + rebar_mass * cost['Rebar']
        total_carbon = steel_mass * lci['Steel'] + concrete_mass * lci['Concrete'] + rebar_mass * lci['Rebar']

        penalty = 0
        if y_pred < target_load:
            penalty += 1000 * (target_load - y_pred)**2
        if y_pred > 1.1 * target_load:
            penalty += 1000 * (y_pred - 1.1 * target_load)**2
        if total_cost > max_cost:
            penalty += 1000 * (total_cost - max_cost)**2
        if total_carbon > max_carbon:
            penalty += 1000 * (total_carbon - max_carbon)**2

        return penalty

    if st.button("Run Optimisation"):
        result = differential_evolution(objective, bounds, maxiter=200, polish=True)
        opt_input = result.x
        X_opt = pd.DataFrame([opt_input], columns=model_feature_names)
        X_opt_scaled = scaler.transform(X_opt)
        y_opt = model.predict(X_opt_scaled)[0]

        st.success("Optimisation completed.")
        st.write("### Optimal Design Variables:")
        for f, v in zip(display_feature_names, opt_input):
            if f == 'Steel Shape':
                shape_name = [k for k, val in shape_mapping.items() if val == int(round(v))][0]
                st.write(f"**{f}**: {shape_name} ({int(round(v))})")
            else:
                st.write(f"**{f}**: {v:.2f}")

        st.write("### Prediction and Constraints")
        st.write(f"**Predicted Load Capacity:** {y_opt:.2f} kN")
        st.write(f"**Prediction Interval (95%):** ({y_opt*0.95:.2f} kN, {y_opt*1.05:.2f} kN)")

        steel_mass, concrete_mass, rebar_mass = opt_input[5], opt_input[4], opt_input[6]
        total_cost = steel_mass * cost['Steel'] + concrete_mass * cost['Concrete'] + rebar_mass * cost['Rebar']
        total_carbon = steel_mass * lci['Steel'] + concrete_mass * lci['Concrete'] + rebar_mass * lci['Rebar']

        st.write(f"**Total Cost:** ${total_cost:.2f}")
        st.write(f"**Total Carbon:** {total_carbon:.2f} kg CO₂")

        plot_feature_importance(model, X_opt_scaled, display_feature_names)

    # Add footnote
    st.markdown("""
        **Notes**: 
        1. This application predicts the ultimate bond strength of FRP-concrete interface using categorical boosting algorithm optimised with Optuna.
        2. The model was trained using data from single-lap shear test experiments.
    """)
    
    st.markdown("""
        **References**: 
        1. L. Prokhorenkova, G. Gusev, A. Vorobev, A.V. Dorogush, A. Gulin, CatBoost: unbiased boosting with categorical features, 2018. https://github.com/catboost/catboost.
        2. T. Akiba, S. Sano, T. Yanase, T. Ohta, M. Koyama, Optuna: A Next-generation Hyperparameter Optimization Framework, in: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, Association for Computing Machinery, New York, NY, USA, 2019: pp. 2623–2631. https://doi.org/10.1145/3292500.3330701.
    """)

# Adding a footer with contact information
footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f1f1f1;
    text-align: center;
    padding: 10px;
    font-size: 12px;
    color: #6c757d;
}
</style>
<div class="footer">
    <p>© 2024 My Streamlit App. All rights reserved. | Temitope E. Dada, Silas E. Oluwadahunsi | For Queries: <a href="mailto: T.Dada19@student.xjtlu.edu.cn"> T.Dada19@student.xjtlu.edu.cn</a></p>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)