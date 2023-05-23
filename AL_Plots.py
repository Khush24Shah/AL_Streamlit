import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def plot_cluster(x, y, model=None, samples=None, name=None):
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
        opacity=0.75,
    ))

    if model is None:
        # Plot data points when no model is provided
        fig.add_trace(go.Scatter(
            x=x[:, 0],
            y=x[:, 1],
            mode='markers',
            marker=dict(color='orange', line=dict(color='black', width=1)),
        ))
    else:
        # Plot data points with labels when a model is provided
        fig.add_trace(go.Scatter(
            x=x[:, 0],
            y=x[:, 1],
            mode='markers',
            marker=dict(color=y, colorscale='RdYlBu', line=dict(color='black', width=1))
        ))

    if samples is not None:
        # Plot additional samples if available
        fig.add_trace(go.Scatter(
            x=samples[:, 0],
            y=samples[:, 1],
            mode='markers',
            marker=dict(color='white', colorscale='RdYlBu', line=dict(color='black', width=1))
        ))

    fig.update_layout(
        xaxis=dict(range=(x_min, x_max)),
        yaxis=dict(range=(y_min, y_max)),
        width=650,
        height=650
    )

    if name:
        fig.update_layout(title=name)
    
    st.write(fig)


def plot_contour(x, model=None):
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


def plot_accuracy(title, overall_accuracy, models_accuracy_list, classifiers, iteration):
    fig = px.line(title=title, labels={"x": "Iterations", "y": "Accuracy"})
    fig.add_scatter(x=np.arange(iteration), y=overall_accuracy, name="Overall")
    for i in range(len(models_accuracy_list)):
        fig.add_scatter(x=np.arange(iteration), y=models_accuracy_list[i], name=classifiers[i])
    return fig