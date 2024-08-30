import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import itertools

from clean_data import clean_data, merge_scope_with_data
from gel_pred import train_gel_model, restrict_scope, predict_gel
from vis_gel import plot_gel_ternary
# import plotly.express as px
# import plotly.graph_objects as go
st.set_page_config(layout="wide")


# Read in initial dataset
data_path = 'Multifunctional thiol data summary 2024_06_14_ML.csv'

# Separate into gel and nogel datasets
gel_df, nogel_df = clean_data(data_path)

# Train gel classification model
gel_model, scaler, feature_importance, detailed_results = train_gel_model(gel_df)

# Define range of reaction conditions to explore
reaction_components = {
    'Total_Conc_M': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    'BDDA_mol%': [0.5] + list(np.arange(1, 31)),
    'Elp_mol%': list(np.linspace(0, 80, 161)),
}

# Create DataFrame from all combinations
all_combinations = list(itertools.product(reaction_components['Total_Conc_M'], reaction_components['BDDA_mol%'], reaction_components['Elp_mol%']))
scope_df = pd.DataFrame(all_combinations, columns=['Total_Conc_M', 'BDDA_mol%', 'Elp_mol%'])
gel_pred_df = predict_gel(gel_model, scaler, scope_df)
# gel_pred_df

gel_pred_df['nBA_mol%'] = 100 - gel_pred_df['Elp_mol%'] - gel_pred_df['BDDA_mol%']
gel_df['nBA_mol%'] = 100 - gel_df['Elp_mol%'] - gel_df['BDDA_mol%']



# Streamlit application
st.title('Predicting Probability of Gelation using Logistic Regression')
c1, c2 = st.columns([5,5])
c = c1.container()
conc = c1.slider('Concentration', min_value=1.0, max_value=2.0, step=0.1, format='%.1f', label_visibility='hidden')
c.markdown(f'#### Concentration: {conc:.1f} (M)')

# c1.divider()
c1.markdown(f'#### Results:')
c11, c12 = c1.columns(2)
c11.write('Logistic Regression Coefficients:')
c11.dataframe(feature_importance, hide_index=True)

c12.write('5-Fold Cross Validation Results:')
c12.dataframe(detailed_results, hide_index=True)

with c2:
    plot_gel_ternary(gel_df, gel_pred_df, conc=conc)
