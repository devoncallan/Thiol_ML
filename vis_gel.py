import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import itertools

from clean_data import clean_data, merge_scope_with_data
from gel_pred import train_gel_model, restrict_scope, predict_gel
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

def plot_gel_ternary(gel_df, gel_pred_df, conc=1.0):
    
    # Prepare your first dataset
    # conc = 1.0
    gel_pred = gel_pred_df[gel_pred_df['Total_Conc_M'] == conc]
    gel = gel_df[gel_df['Total_Conc_M'] == conc]

    # Define a hover template
    hovertemplate_exp0 = ('BDDA: %{a:.1f} mol%<br>' +
                    'Elp: %{b:.1f} mol%<br>' +
                    'nBA: %{c:.1f} mol%<br>' +
                    'Gel(Exp): False<extra></extra>')  # <extra></extra> avoids displaying trace name
    
    hovertemplate_exp1 = ('BDDA: %{a:.1f} mol%<br>' +
                'Elp: %{b:.1f} mol%<br>' +
                'nBA: %{c:.1f} mol%<br>' +
                'Gel(Exp): True<extra></extra>')  # <extra></extra> avoids displaying trace name

    hovertemplate_pred = ('BDDA: %{a:.1f} mol%<br>' +
                    'Elp: %{b:.1f} mol%<br>' +
                    'nBA: %{c:.1f} mol%<br>' +
                    'P(Gel): %{marker.color:.2f}<extra></extra>')  # <extra></extra> avoids displaying trace name

    fig = go.Figure(go.Scatterternary({
        'mode': 'markers',
        'a': gel_pred['BDDA_mol%'],
        'b': gel_pred['Elp_mol%'],
        'c': gel_pred['nBA_mol%'],
        'marker': {
            'symbol': 'circle', 
            'color': gel_pred['Gel'], 
            'colorscale': 'RdBu', 
            # 'coloraxis': 'coloraxis',
            'showscale': True,
            'cmin': 0, 
            'cmax': 1, 
            'size': 5, 
            'opacity': 0.15,  # Adjust opacity here
            
        },
        'name': 'Gel (pred)',
        'hovertemplate': hovertemplate_pred
    }))
    
    # Filter data within the specified range around 0.5
    n = 0.05
    threshold_min = 0.5 - n
    threshold_max = 0.5 + n
    gel_pred = gel_pred_df[(gel_pred_df['Total_Conc_M'] == conc) & 
                           (gel_pred_df['Gel'].between(threshold_min, threshold_max))]
    
    if len(gel_pred) > 2:

        # Fit a PCA to find the main direction of these data points
        pca = PCA(n_components=1)
        components = pca.fit_transform(gel_pred[['BDDA_mol%', 'Elp_mol%', 'nBA_mol%']])
        direction = pca.components_[0]
        
        # Project points along the principal component to find min and max projections
        projected = np.dot(gel_pred[['BDDA_mol%', 'Elp_mol%', 'nBA_mol%']], direction)
        min_idx, max_idx = projected.argmin(), projected.argmax()

        # Endpoints of the line
        point_min = gel_pred.iloc[min_idx][['BDDA_mol%', 'Elp_mol%', 'nBA_mol%']].values
        point_max = gel_pred.iloc[max_idx][['BDDA_mol%', 'Elp_mol%', 'nBA_mol%']].values

        # Plotting
        fig.add_trace(go.Scatterternary({
            'mode': 'lines',
            'a': [point_min[0], point_max[0]],
            'b': [point_min[1], point_max[1]],
            'c': [point_min[2], point_max[2]],
            'line': {'color': 'black', 'width': 2},
            'name': 'Predicted Boundary'
        }))

    # Add the second dataset to the plot with its own hovertemplate
    gel0 = gel[gel['Gel'] == 0]
    gel1 = gel[gel['Gel'] == 1]


    fig.add_trace(go.Scatterternary({
        'mode': 'markers',
        'a': gel0['BDDA_mol%'],
        'b': gel0['Elp_mol%'],
        'c': gel0['nBA_mol%'],
        'marker': {
            'symbol': 'x', 
            'color': 'red',
            'size': 5
        },
        'name': 'No Gel (exp)',
        'hovertemplate': hovertemplate_exp0
    }))

    fig.add_trace(go.Scatterternary({
        'mode': 'markers',
        'a': gel1['BDDA_mol%'],
        'b': gel1['Elp_mol%'],
        'c': gel1['nBA_mol%'],
        'marker': {
            'symbol': 'x', 
            'color': 'blue',
            'size': 5
        },
        'name': 'Gel (exp)',
        'hovertemplate': hovertemplate_exp1
    }))

    # Update layout if necessary
    fig.update_layout({
        'coloraxis': {
            'colorbar': {
                # 'color': 'RdBu',
                'title': 'Probability of Gel',  # Colorbar title
                'titleside': 'top'
            }
        },
        'ternary': {
            'sum': 100,
            'aaxis': {'title': 'BDDA_mol%', 'min': 0.01},
            'baxis': {'title': 'Elp_mol%', 'min': 0.01},
            'caxis': {'title': 'nBA_mol%', 'min': 0.01},
        },
        'legend': {'x': 0.1, 'y': 1.1},  # Adjust the position of the legend
        # 'coloraxis_colorbar': {
        #     'title': 'Probability of Gel',  # Set the title for the colorbar
        #     'titleside': 'top'
        # },
        # 'coloraxis_colorbar': {
        #     'x': 0.05,  # Adjust the position of the colorbar
        #     'y': 0.5,
        #     'len': 0.4  # Optionally adjust the length of the colorbar
        # }
})

    # Show the figure
    st.plotly_chart(fig)
    # fig.show()
    
