import torch
import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import streamlit as st

def plot_cluster(x, y, model=None, samples=None, name=None):
    # x_min, x_max =x[:, 0].min(), x[:, 0].max() 
    # y_min, y_max =x[:, 1].min(), x[:, 1].max() 

    # xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.05), torch.arange(y_min, y_max, 0.05))

    # if model is None:
    #     Z = torch.empty(xx.ravel().shape)
    #     for i in range(xx.ravel().shape[0]):
    #         if (xx.ravel()[i] * yy.ravel()[i] > torch.tensor(0)):
    #             Z[i]=1
    #         else:
    #             Z[i]=0
    # else:
    #     Z=model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Z=Z.reshape(xx.shape)
    # ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    # if model is None:
    #     ax.scatter(x, y, color="orange")
    # else:
    #     ax.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis')
    
    # if samples is not None:
    #     ax.scatter(samples[:, 0], samples[:, 1], color="white", cmap="viridis")
    
    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)

    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

    if model is None:
        Z = np.empty(xx.ravel().shape)
        for i in range(xx.ravel().shape[0]):
            Z[i] = 1 if xx.ravel()[i] * yy.ravel()[i] > 0 else 0
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    fig = go.Figure()

    # Plot decision boundaries
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, 0.05),
        y=np.arange(y_min, y_max, 0.05),
        z=Z,
        colorscale='RdYlBu',
        contours_coloring='heatmap',
        opacity=1
    ))

    if model is None:
        # Plot data points when no model is provided
        fig.add_trace(go.Scatter(
            x=x[:, 0],
            y=x[:, 1],
            mode='markers',
            marker=dict(color='orange')
        ))
    else:
        # Plot data points with labels when a model is provided
        fig.add_trace(go.Scatter(
            x=x[:, 0],
            y=x[:, 1],
            mode='markers',
            marker=dict(color=y, colorscale='viridis')
        ))

    if samples is not None:
        # Plot additional samples if available
        fig.add_trace(go.Scatter(
            x=samples[:, 0],
            y=samples[:, 1],
            mode='markers',
            marker=dict(color='white', colorscale='viridis')
        ))

    fig.update_layout(
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max])
    )

    if name:
        fig.update_layout(title=name)

    st.write("Here")
    st.write(fig)


def plot_contour(x, model=None):
    # x_min, x_max =x[:, 0].min(), x[:, 0].max() 
    # y_min, y_max =x[:, 1].min(), x[:, 1].max()

    # xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.05), torch.arange(y_min, y_max, 0.05))

    # if model is None:
    #     Z = torch.empty(xx.ravel().shape)
    #     for i in range(xx.ravel().shape[0]):
    #         Z[i] = 1 if (xx.ravel()[i] * yy.ravel()[i] > torch.tensor(0)) else 0
    # else:
    #     Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Z = Z.reshape(xx.shape)

    # return (xx, yy, Z)

    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

    if model is None:
        Z = np.empty(xx.ravel().shape)
        for i in range(xx.ravel().shape[0]):
            Z[i] = 1 if xx.ravel()[i] * yy.ravel()[i] > 0 else 0
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    return (xx, yy, Z)