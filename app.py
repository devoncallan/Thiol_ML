import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import itertools
from scipy.interpolate import griddata

from clean_data import clean_data, merge_scope_with_data
from gel_pred import train_gel_model, restrict_scope, predict_gel
from vis_gel import plot_gel_ternary, plot_prediction_surface, plot_gel_ternary_grid

from process_data import get_logistic_regression_data
# import plotly.express as px
# import plotly.graph_objects as go
st.set_page_config(layout="wide")


# Read in initial dataset
gel_df, gel_pred_df, results = get_logistic_regression_data()
feature_importance, detailed_results = results

# Streamlit application
st.title('Predicting Probability of Gelation using Logistic Regression')
c1, c2 = st.columns([5,5])
c1L, c1R = c1.columns(2)

vis_types = ['Ternary Plot', 'Prediction Surface']
c1L.markdown('#### Visualization')
selected_vis = c1L.radio('Visualization:', vis_types, label_visibility='collapsed')



# conc = c1.slider('Concentration', min_value=1.0, max_value=2.0, step=0.1, format='%.1f', label_visibility='hidden')
if selected_vis == 'Ternary Plot':
    c1R_ = c1R.container()
    conc = c1R.number_input('Concentration', min_value=1.0, max_value=2.0, step=0.1, format='%.1f', label_visibility='collapsed')
    conc = np.round(conc, 1)
    c1R_.markdown(f'#### Concentration: {conc:.1f} (M)')




c1.divider()
c1.markdown(f'#### Results:')
c11, c12 = c1.columns(2)
c11.write('Logistic Regression Coefficients:')
c11.dataframe(feature_importance, hide_index=True)

c12.write('5-Fold Cross Validation Results:')
c12.dataframe(detailed_results, hide_index=True)

with c2:
    if selected_vis == 'Ternary Plot':
        plot_gel_ternary(gel_df, gel_pred_df, conc=conc)
    else:
        plot_prediction_surface(gel_df, gel_pred_df)
