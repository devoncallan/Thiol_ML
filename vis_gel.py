import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import interp1d
import itertools

from clean_data import clean_data, merge_scope_with_data
from gel_pred import train_gel_model, restrict_scope, predict_gel
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plotly.subplots import make_subplots
import plotly.graph_objects as go

NUM_ROWS = 3
NUM_COLS = 4
HEIGHT = 1100
CONCENTRATIONS = np.round(np.linspace(1.0, 2.0, 11), 1)

def add_gel_prediction_trace(fig, gel_pred, row, col, symbol='circle', color_scale='RdBu', size=5, opacity=0.15):
    fig.add_trace(go.Scatterternary({
        'mode': 'markers',
        'a': gel_pred['BDDA_mol%'],
        'b': gel_pred['Elp_mol%'],
        'c': gel_pred['nBA_mol%'],
        'marker': {
            'symbol': symbol,
            'color': gel_pred['Uncertainty'],
            'colorscale': color_scale,
            'showscale': row == 1 and col == 1,  # Show scale only for the first subplot
            'cmin': 0,
            'cmax': 1,
            'size': size,
            'opacity': opacity
        },
        'name': 'Gel (pred)',
        'hovertemplate': ('BDDA: %{a:.1f} mol%<br>' +
                          'Elp: %{b:.1f} mol%<br>' +
                          'nBA: %{c:.1f} mol%<br>' +
                          'P(Gel): %{marker.color:.2f}<extra></extra>')
    }), row=row, col=col)



def create_ternary_grid():
    # Create a subplot grid with 'ternary' specifications
    fig = make_subplots(
        rows=NUM_ROWS, cols=NUM_COLS,
        subplot_titles=[f'Concentration: {conc:.1f} M' for conc in CONCENTRATIONS],
        specs=[[{'type': 'ternary'} for _ in range(NUM_COLS)] for _ in range(NUM_ROWS)],
        horizontal_spacing=0.05, vertical_spacing=0.1,
        
    )
    return fig
    
def add_points(fig, data, symbol='x', color='red', size=5, colorscale='RdBu'):
    
    for i, conc in enumerate(CONCENTRATIONS):
        row = i // NUM_COLS + 1
        col = i % NUM_COLS + 1
        
        # Filter data for current concentration
        data_at_conc = data[data['Total_Conc_M'] == conc]
        
        if color in data.columns:
            c = data_at_conc[color]
        else:
            c = color
        
        fig.add_trace(go.Scatterternary({
            'mode': 'markers',
            'a': data_at_conc['BDDA_mol%'],
            'b': data_at_conc['Elp_mol%'],
            'c': data_at_conc['nBA_mol%'],
            'marker': {
                'symbol': symbol,
                'color': c,
                'size': size,
                'colorscale': colorscale,
            },
            
            # 'name': name,
            
            'hovertemplate': ('BDDA: %{a:.1f} mol%<br>' +
                            'Elp: %{b:.1f} mol%<br>' +
                            'nBA: %{c:.1f} mol%<br>')
                            #   f'Gel(Exp): {hover_text}<extra></extra>')
        }), row=row, col=col)
        
        fig.update_layout({
            f'ternary{i+1}': {
                'sum': 100,
                'aaxis': {'title': {'text': 'BDDA'}, 'min': 0.01},
                'baxis': {'title': {'text': 'Elp'}, 'min': 0.01},
                'caxis': {'title': {'text': 'nBA'}, 'min': 0.01},
            },
            
            'showlegend': False
        })
        
    return fig
        
def set_ternary_axes(fig):
    
    fig.update_layout(
        height=HEIGHT,
        coloraxis_colorbar=dict(
            title='Probability of Gel',  # Colorbar title
            titleside='top'
        ),
        
    )
    return fig
    


def plot_gel_ternary_grid(gel_df, gel_pred_df, concentrations, rows=4, cols=3):
    # Create a subplot grid with 'ternary' specifications
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f'Concentration: {conc:.1f} M' for conc in concentrations],
        specs=[[{'type': 'ternary'} for _ in range(cols)] for _ in range(rows)],
        horizontal_spacing=0.05, vertical_spacing=0.1
    )

    for i, conc in enumerate(concentrations):
        row = i // cols + 1
        col = i % cols + 1

        # Filter data for current concentration
        gel_pred = gel_pred_df[gel_pred_df['Total_Conc_M'] == conc]
        gel = gel_df[gel_df['Total_Conc_M'] == conc]
        
        gel0 = gel[gel['Gel'] == 0]
        gel1 = gel[gel['Gel'] == 1]

        # Add prediction and experimental data traces
        add_gel_prediction_trace(fig, gel_pred, row, col)
        add_points_ternary(fig, gel0, row, col, gel_value=0, color='red')
        add_points_ternary(fig, gel1, row, col, gel_value=1, color='blue')

    # Update layout for all plots
    fig.update_layout(
        showlegend=False  # Turn off legend for cleaner subplots
    )
    
    # Show the figure
    fig.show()


def plot_gel_ternary_grid(gel_df, gel_pred_df, concentrations, rows=4, cols=3):
    # Create a subplot grid with the correct 'ternary' specifications
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f'Concentration: {conc:.1f} M' for conc in concentrations],
        specs=[[{'type': 'ternary'} for _ in range(cols)] for _ in range(rows)],
        # horizontal_spacing=0.05, vertical_spacing=0.2
    )

    for i, conc in enumerate(concentrations):
        # Prepare the data for the current concentration
        row = i // cols + 1
        col = i % cols + 1

        gel_pred = gel_pred_df[gel_pred_df['Total_Conc_M'] == conc]
        gel = gel_df[gel_df['Total_Conc_M'] == conc]

        # Adding prediction scatter plot
        fig.add_trace(go.Scatterternary({
            'mode': 'markers',
            'a': gel_pred['BDDA_mol%'],
            'b': gel_pred['Elp_mol%'],
            'c': gel_pred['nBA_mol%'],
            'marker': {
                'symbol': 'circle', 
                # 'color': gel_pred['Gel'], 
                'color': gel_pred['Uncertainty'],
                'colorscale': 'RdBu', 
                'showscale': i == 0,  # Show scale only for the first plot
                'cmin': 0, 
                'cmax': 1, 
                'size': 5, 
                'opacity': 0.15
            },
            'name': 'Gel (pred)',
            'hovertemplate': ('BDDA: %{a:.1f} mol%<br>' +
                              'Elp: %{b:.1f} mol%<br>' +
                              'nBA: %{c:.1f} mol%<br>' +
                              'P(Gel): %{marker.color:.2f}<extra></extra>')
        }), row=row, col=col)

        # Add experimental data
        gel0 = gel[gel['Gel'] == 0]
        gel1 = gel[gel['Gel'] == 1]

        fig.add_trace(go.Scatterternary({
            'mode': 'markers',
            'a': gel0['BDDA_mol%'],
            'b': gel0['Elp_mol%'],
            'c': gel0['nBA_mol%'],
            'marker': {'symbol': 'x', 'color': 'red', 'size': 5},
            'name': 'No Gel (exp)',
            'hovertemplate': ('BDDA: %{a:.1f} mol%<br>' +
                              'Elp: %{b:.1f} mol%<br>' +
                              'nBA: %{c:.1f} mol%<br>' +
                              'Gel(Exp): False<extra></extra>')
        }), row=row, col=col)

        fig.add_trace(go.Scatterternary({
            'mode': 'markers',
            'a': gel1['BDDA_mol%'],
            'b': gel1['Elp_mol%'],
            'c': gel1['nBA_mol%'],
            'marker': {'symbol': 'x', 'color': 'blue', 'size': 5},
            'name': 'Gel (exp)',
            'hovertemplate': ('BDDA: %{a:.1f} mol%<br>' +
                              'Elp: %{b:.1f} mol%<br>' +
                              'nBA: %{c:.1f} mol%<br>' +
                              'Gel(Exp): True<extra></extra>')
        }), row=row, col=col)

        # Update axes labels for each ternary subplot by specifying 'ternary' + index
        fig.update_layout({
            f'ternary{i+1}': {
                'sum': 100,
                'aaxis': {'title': {'text': 'BDDA', 'font': {'lineposition': 'under'}}, 'min': 0.01},
                'baxis': {'title': {'text': 'Elp'}, 'min': 0.01},
                'caxis': {'title': {'text': 'nBA'}, 'min': 0.01},
            },
            'showlegend': False
        })

    # Adjust layout to display the colorbar only for the first subplot
    fig.update_layout(
        height=1000,
        coloraxis_colorbar=dict(
            title='Probability of Gel',  # Colorbar title
            titleside='top'
        
    ))
    fig.update_layout(coloraxis_colorbar=dict(
        title='Probability of Gel',  # Colorbar title
        titleside='top'
        
    ))

    # Show the figure
    # fig.show()
    st.plotly_chart(fig)



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
    # n = 0.05
    # threshold_min = 0.5 - n
    # threshold_max = 0.5 + n
    # gel_pred = gel_pred_df[(gel_pred_df['Total_Conc_M'] == conc) & 
    #                        (gel_pred_df['Gel'].between(threshold_min, threshold_max))]
    
    # if len(gel_pred) > 2:

    #     # Fit a PCA to find the main direction of these data points
    #     pca = PCA(n_components=1)
    #     components = pca.fit_transform(gel_pred[['BDDA_mol%', 'Elp_mol%', 'nBA_mol%']])
    #     direction = pca.components_[0]
        
    #     # Project points along the principal component to find min and max projections
    #     projected = np.dot(gel_pred[['BDDA_mol%', 'Elp_mol%', 'nBA_mol%']], direction)
    #     min_idx, max_idx = projected.argmin(), projected.argmax()

    #     # Endpoints of the line
    #     point_min = gel_pred.iloc[min_idx][['BDDA_mol%', 'Elp_mol%', 'nBA_mol%']].values
    #     point_max = gel_pred.iloc[max_idx][['BDDA_mol%', 'Elp_mol%', 'nBA_mol%']].values

    #     # Plotting
    #     fig.add_trace(go.Scatterternary({
    #         'mode': 'lines',
    #         'a': [point_min[0], point_max[0]],
    #         'b': [point_min[1], point_max[1]],
    #         'c': [point_min[2], point_max[2]],
    #         'line': {'color': 'black', 'width': 2},
    #         'name': 'Predicted Boundary'
    #     }))

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
    
# Function to interpolate Total_Conc_M at Gel == 0.5 for each group
def interpolate_gel_05(group):
    # Sort group by Total_Conc_M
    group = group.sort_values('Total_Conc_M')
    
    # Only interpolate if Gel spans 0.5 within this group
    if group['Gel'].min() <= 0.5 <= group['Gel'].max():
        # Linear interpolation
        interp_func = interp1d(group['Gel'], group['Total_Conc_M'], kind='linear')
        return interp_func(0.5)
    else:
        return np.nan
    
def find_terraced_gel_05(group):
    # Sort group by Total_Conc_M to maintain order
    group = group.sort_values('Total_Conc_M')
    
    # Find the row where Gel is closest to 0.5
    closest_idx = (group['Gel'] - 0.5).abs().idxmin()
    return group.loc[closest_idx, 'Total_Conc_M']

# def plot_prediction_surface(gel_df, gel_pred_df):
    
#     # Apply the interpolation function to each BDDA and Elp group
#     surface_data = gel_pred_df.groupby(['BDDA_mol%', 'Elp_mol%']).apply(interpolate_gel_05).reset_index()
#     surface_data.columns = ['BDDA_mol%', 'Elp_mol%', 'Total_Conc_M']

#     # Drop NaN values (combinations where interpolation was not possible)
#     surface_data = surface_data.dropna()

#     # Pivot to create a grid for surface plotting
#     z_values = surface_data.pivot(index='Elp_mol%', columns='BDDA_mol%', values='Total_Conc_M')
#     x_values = z_values.columns.values
#     y_values = z_values.index.values

#     # Create the surface plot
#     fig = go.Figure(data=[go.Surface(
#         x=x_values,
#         y=y_values,
#         z=z_values.values,
#         colorscale="Viridis",
#         colorbar=dict(title="Total_Conc_M at Gel ≈ 0.5")
#     )])

#     # Update layout for clarity
#     fig.update_layout(
#         scene=dict(
#             xaxis_title="BDDA_mol%",
#             yaxis_title="Elp_mol%",
#             zaxis_title="Total_Conc_M"
#         ),
#         title="Surface Plot of Total Concentration for Gel ≈ 0.5"
#     )

#     st.plotly_chart(fig)

def plot_prediction_surface(gel_df, gel_pred_df):
    
    # Apply the interpolation function to each BDDA and Elp group
    surface_data = gel_pred_df.groupby(['BDDA_mol%', 'Elp_mol%']).apply(interpolate_gel_05).reset_index()
    surface_data.columns = ['BDDA_mol%', 'Elp_mol%', 'Total_Conc_M']

    # Drop NaN values (combinations where interpolation was not possible)
    surface_data = surface_data.dropna()

    # Pivot to create a grid for surface plotting
    z_values = surface_data.pivot(index='Elp_mol%', columns='BDDA_mol%', values='Total_Conc_M')
    x_values = z_values.columns.values
    y_values = z_values.index.values

    # Create the surface plot
    fig = go.Figure(data=[go.Surface(
        x=x_values,
        y=y_values,
        z=z_values.values+0.001,
        # surfacecolor=np.ones_like(z_values.values),  # Uniform color across the surface
        opacity=1.0,  # Set transparency level
        colorscale=[[0, '#AEC6CF'], [1, '#AEC6CF']],  # Single color: black
        showscale=False  # Hide color scale
    )])
    
    fig.add_trace(go.Surface(
        x=x_values,
        y=y_values,
        z=z_values.values-0.001,
        # surfacecolor=np.ones_like(z_values.values),  # Uniform color across the surface
        opacity=1.0,  # Set transparency level
        colorscale=[[0, '#F4C2C2'], [1, '#F4C2C2']],  # Single color: black
        showscale=False  # Hide color scale
    ))

    # Separate the experimental data into gel and no-gel points
    gel0 = gel_df[gel_df['Gel'] == 0]  # No gel
    gel1 = gel_df[gel_df['Gel'] == 1]  # Gel

    # Add scatter plot for no-gel points (red 'o' markers)
    fig.add_trace(go.Scatter3d(
        x=gel0['BDDA_mol%'],
        y=gel0['Elp_mol%'],
        z=gel0['Total_Conc_M'],
        mode='markers',
        marker=dict(symbol='circle', color='red', size=5),
        name='No Gel (exp)'
    ))

    # Add scatter plot for gel points (blue 'x' markers)
    fig.add_trace(go.Scatter3d(
        x=gel1['BDDA_mol%'],
        y=gel1['Elp_mol%'],
        z=gel1['Total_Conc_M'],
        mode='markers',
        marker=dict(symbol='x', color='blue', size=5),
        name='Gel (exp)'
    ))

    # Update layout for clarity
    fig.update_layout(
        width=1000,
        height=800,
        scene=dict(
            xaxis_title="BDDA_mol%",
            yaxis_title="Elp_mol%",
            zaxis_title="Total_Conc_M"
        ),
        title="Surface Plot of Total Concentration for Gel ≈ 0.5 with Experimental Data"
    )

    st.plotly_chart(fig)