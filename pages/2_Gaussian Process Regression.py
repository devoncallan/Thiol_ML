import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import itertools
from scipy.interpolate import griddata

from clean_data import clean_data, merge_scope_with_data
from gel_pred import train_gel_model, restrict_scope, predict_gel
from vis_gel import plot_gel_ternary, plot_prediction_surface, plot_gel_ternary_grid, create_ternary_grid, add_points, set_ternary_axes
from process_data import get_logistic_regression_data, get_gel_data
st.set_page_config(layout="wide")


@st.cache_data()
def read_gp_pred():
    return pd.read_csv('GP_Pred.csv')

# Streamlit application
st.title('Gaussian Process Regression')

gel_df, nogel_df = get_gel_data(all=True)

pred_df = read_gp_pred()

concentrations = np.round(np.linspace(1.0, 2.0, 11), 1)
# plot_gel_ternary_grid(gel_df, pred_df, concentrations, rows=3, cols=4)

# Visualize Prediction or Uncertainty
c1, c2 = st.columns(2)
c1.markdown('#### Visualize')
selected_vis = c1.radio('Visualization:', ['Prediction', 'Uncertainty'], label_visibility='collapsed')


t_samples = pd.read_csv('Thompson_samples.csv')
t_samples['nBA_mol%'] = 100 - t_samples['Elp_mol%'] - t_samples['BDDA_mol%']
t_samples = t_samples[t_samples['nBA_mol%'] >= 0]

bo_samples = pd.read_csv('BO_samples.csv')
bo_samples['nBA_mol%'] = 100 - bo_samples['Elp_mol%'] - bo_samples['BDDA_mol%']
bo_samples = bo_samples[bo_samples['nBA_mol%'] >= 0]

fig = create_ternary_grid()
if selected_vis == 'Prediction':
    fig = add_points(fig, pred_df, 'x', color='Gel', size=5)
else:
    fig = add_points(fig, pred_df, 'x', color='Uncertainty', size=5)
fig = add_points(fig, gel_df[gel_df['Gel'] == 0], 'x', color='red')
fig = add_points(fig, gel_df[gel_df['Gel'] == 1], 'x', color='blue')
fig = add_points(fig, t_samples, 'x', color='green', size=5)
fig = add_points(fig, bo_samples, 'x', color='orange', size=5)
fig = set_ternary_axes(fig)
st.plotly_chart(fig)