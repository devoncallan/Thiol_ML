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

@st.cache_data()
def get_logistic_regression_data():
    
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
    
    results = [feature_importance, detailed_results]
    
    return gel_df, gel_pred_df, results

def get_gel_data(all=False):
    
    # Read in initial dataset
    data_path = 'Multifunctional thiol data summary 2024_06_14_ML.csv'

    # Separate into gel and nogel datasets
    gel_df, nogel_df = clean_data(data_path)
    
    if all:
        gel_df['nBA_mol%'] = 100 - gel_df['Elp_mol%'] - gel_df['BDDA_mol%']
        nogel_df['nBA_mol%'] = 100 - nogel_df['Elp_mol%'] - nogel_df['BDDA_mol%']
    
    return gel_df, nogel_df