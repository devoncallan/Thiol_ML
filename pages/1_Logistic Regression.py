import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import itertools
from scipy.interpolate import griddata

from clean_data import clean_data, merge_scope_with_data
from gel_pred import train_gel_model, restrict_scope, predict_gel
# from vis_gel import plot_gel_ternary, plot_prediction_surface, plot_gel_ternary_grid
from vis_gel import plot_gel_ternary, plot_prediction_surface, plot_gel_ternary_grid, create_ternary_grid, add_points, set_ternary_axes

from process_data import get_logistic_regression_data
# import plotly.express as px
# import plotly.graph_objects as go
st.set_page_config(layout="wide")


# Read in initial dataset

gel_df, gel_pred_df, results = get_logistic_regression_data()
feature_importance, detailed_results = results
# gel_model, scaler, feature_importance, detailed_results = train_gel_model(gel_df)




# Streamlit application
st.title('Logistic Regression')
# c1, c2 = st.columns([5,5])
# c = c1.container()
# conc = c1.slider('Concentration', min_value=1.0, max_value=2.0, step=0.1, format='%.1f', label_visibility='hidden')
# c.markdown(f'#### Concentration: {conc:.1f} (M)')

# c1.divider()
# c1.markdown(f'#### Results:')
# c11, c12 = c1.columns(2)
# c11.write('Logistic Regression Coefficients:')
# c11.dataframe(feature_importance, hide_index=True)

# c12.write('5-Fold Cross Validation Results:')
# c12.dataframe(detailed_results, hide_index=True)

# with c2:
    # plot_gel_ternary(gel_df, gel_pred_df, conc=conc)
    # plot_prediction_surface(gel_df, gel_pred_df)
fig = create_ternary_grid()
fig = add_points(fig, gel_pred_df, 'x', color='Gel', size=5)
fig = add_points(fig, gel_df[gel_df['Gel'] == 0], 'x', color='red')
fig = add_points(fig, gel_df[gel_df['Gel'] == 1], 'x', color='blue')
fig = set_ternary_axes(fig)
st.plotly_chart(fig)
