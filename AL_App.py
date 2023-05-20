import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score
from AL_Metrics import get_uncertainty
from AL_Plots import plot_cluster
from AL_Animation import plotAnimation

def validate_size(classifiers, size=0):
    s = len(classifiers)

    if s == 0:
        return False
    
    if size != 0 and s != size:
        return False
    return True

# Title
st.title("Active Learning Calssification")

datasets = {
    "Moons": make_moons(n_samples=1000, noise=0.3, random_state=10),
    "Circles": make_circles(n_samples=1000, noise=0.3, factor=0.5, random_state=10),
    "Classification": make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=10),
    "Blobs": make_blobs(n_samples=1000, n_features=2, centers=5, cluster_std=1.75, random_state=10)}

dataset = st.selectbox(label="Select Dataset to train Active Learning Model on:", options=("Moons", "Circles", "Classification", "Blobs"))

st.write(f"You selected: {dataset}")

x, y = datasets[dataset][0], datasets[dataset][1]

# Plot Dataset
fig = px.scatter(x=x[:, 0], y=x[:, 1], color=y, title="Dataset", range_x=(x[:, 0].min() - 0.5, x[:, 0].max() + 0.5), range_y=(x[:, 1].min() - 0.5, x[:, 1].max() + 0.5), labels={"x": "X", "y": "Y"})
st.write(fig)

# Number of samples to add in each iteration
samples = st.slider(label="Choose the number of points to add in each iteration of Active Learning:", min_value=1, max_value=10, value=5)

# Uncertainty Metric
metric = st.selectbox(label="Select Uncertainty Metric", options=("Least Confidence", "Margin Sampling", "Entropy"))
st.write(f"You selected: {metric}")

# Number of Iterations
iterations = st.slider(label="Input the number of Iterations:", min_value=1, max_value=int((0.975)*(1-0.2)*y.shape[0])//samples, value=((int((0.975)*(1-0.2)*y.shape[0])//samples)//20)*10)

# Classifiers
classifier_options = ("Logistic Regression", "K-nearest neighbours", "Decision Tree", "Random Forest", "Gaussian Process")

size = 0

classifiers = st.multiselect("Select at least 1 option:" if size == 0 else f"Select exactly {size} options:", classifier_options)

if validate_size(classifiers, size):
    st.success(f"You have selected the following classifiers: {', '.join(classifiers)}")
    valid = True
else:
    st.error("Please select at least 1 option!" if size == 0 else f"Please select exactly {size} options!")
    valid = False

if not valid:
    st.error("The selections aren't valid. You aren't allowed to proceed further!")
elif st.button("Run Active Learning"):
    classifiers_objects = {"Logistic Regression": LogisticRegression(), "K-nearest neighbours": KNeighborsClassifier(), "Decision Tree": DecisionTreeClassifier(), "Random Forest": RandomForestClassifier(), "Gaussian Process": GaussianProcessClassifier()}
    models = [classifiers_objects[classifier] for classifier in classifiers]

    models_accuracy_list = [[] for _ in range(len(models))]
    models_test_accuracy_list = [[] for _ in range(len(models))]
    overall_accuracy = []
    overall_test_accuracy = []

    anim_model = [plotAnimation(x, y, model=models[i], model_name=classifiers[i]) for i in range(len(models))]
    
    def select_samples(uncertainty, samples):
        return np.argsort(uncertainty)[-samples:]
    
    def output_list(stack):
        output_list = np.array([int(torch.bincount(x).argmax()) for x in stack])
        return output_list
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
    x_training, x_pool, y_training, y_pool = train_test_split(x_train, y_train, test_size=0.975, random_state=10)

    for iteration in range(iterations):
        y_pred = []
        y_pred_train = []
        y_pred_test = []
        for model_idx, model in enumerate(models):
            model.fit(x_training, y_training)
            pred_train = model.predict(x_train)
            models_accuracy_list[model_idx].append(accuracy_score(y_train, pred_train))
            pred_test = model.predict(x_test)
            models_test_accuracy_list[model_idx].append(accuracy_score(y_test, pred_test))
            y_pred.append(torch.tensor(model.predict(x_pool)))
            y_pred_train.append(torch.tensor(pred_train))
            y_pred_test.append(torch.tensor(pred_test))
        
        stack = torch.stack(y_pred, dim = -1)
        sample_idx = select_samples(get_uncertainty(stack, metric=metric), samples)

        stack_train = torch.stack(y_pred_train, dim = -1)

        stack_test = torch.stack(y_pred_test, dim = -1)

        output_train = output_list(stack_train)
        overall_accuracy.append(accuracy_score(output_train, y_train))
        output_test = output_list(stack_test)
        overall_test_accuracy.append(accuracy_score(output_test, y_test))

        for i in range(len(anim_model)):
            anim_model[i].update(x_training, y_training, x_pool[sample_idx], models_accuracy_list[i][iteration], models_test_accuracy_list[i][iteration])

        x_training = np.append(x_training, x_pool[sample_idx], axis=0)
        y_training = np.append(y_training, y_pool[sample_idx], axis=0)

        x_pool = np.delete(x_pool, sample_idx, axis=0)
        y_pool = np.delete(y_pool, sample_idx, axis=0)

        if iteration%(np.ceil(iterations/10)) == 0:
            st.write(f"Completed Iteration: {iteration} with Overall Train Accuracy: {overall_accuracy[-1]} and Overall Test Accuracy: {overall_test_accuracy[-1]}")

    # Plot Training Accuracy
    fig = px.line(title="Training Accuracy", labels={"x": "Iterations", "y": "Accuracy"})
    fig.add_scatter(x=np.arange(iterations), y=overall_accuracy, name="Overall")
    for i in range(len(models_accuracy_list)):
        fig.add_scatter(x=np.arange(iterations), y=models_accuracy_list[i], name=classifiers[i])
    st.write(fig)

    # Plot Test Accuracy
    fig = px.line(title="Test Accuracy", labels={"x": "Iterations", "y": "Accuracy"})
    fig.add_scatter(x=np.arange(iterations), y=overall_test_accuracy, name="Overall")
    for i in range(len(models_test_accuracy_list)):
        fig.add_scatter(x=np.arange(iterations), y=models_test_accuracy_list[i], name=classifiers[i])
    st.write(fig)

    for i in range(len(models)):
        plot_cluster(x, y, model=models[i], name=classifiers[i] + " Final Model")
    
    for i in range(len(anim_model)):
        anim_model[i].show()