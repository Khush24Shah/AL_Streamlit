from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import matplotlib.animation as animation
import streamlit.components.v1 as components

from AL_Plots import plot_contour

class plotAnimation:
    def __init__(self, x, y, model=None, model_name=None):
        self.x_whole = x
        self.y_whole = y
        self.model = model
        self.model_name = model_name

        self.xlim = (x[:, 0].min() - 0.5, x[:, 0].max() + 0.5)
        self.ylim = (x[:, 1].min() - 0.5, x[:, 1].max() + 0.5)

        self.data = []
        self.contour = []

    def update(self, x, y, samples, train_acc, test_acc):
        self.data.append((x, y, samples, train_acc, test_acc))
        self.contour.append(plot_contour(self.x_whole, self.model))
    
    def animate(self, i):
        self.ax.contourf(self.contour[i][0], self.contour[i][1], self.contour[i][2], cmap=plt.cm.RdYlBu)

        if self.model is None:
            self.ax.scatter(self.x_whole, self.y_whole, color="orange")
        else:
            scat = self.ax.scatter(self.data[i][0][:, 0], self.data[i][0][:, 1], c=self.data[i][1], cmap="viridis")

        legend1 = self.ax.legend(*scat.legend_elements(), loc = "upper right", title = "Classes")
        
        self.ax.add_artist(legend1)

        scat = self.ax.scatter(self.data[i][2][:, 0], self.data[i][2][:, 1], color="white", cmap="viridis", label="Last Batch")
        
        self.ax.legend([scat], ["Last Batch"], loc="lower right")

        self.ax.set_title(f"Iteration: {i}, Train Accuracy: {round(self.data[i][3], 3)}, Test Accuracy: {round(self.data[i][4], 3)}")

        return self.ax

    def show(self):
        self.fig, self.ax = plt.subplots()

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)

        if self.model_name:
            self.ax.set_title(self.model_name)

        anim = animation.FuncAnimation(self.fig, self.animate, frames=len(self.data), interval=500, blit=False, repeat=False)
        # video = anim.to_html5_video()

        components.html(anim.to_jshtml(), height=1000)

        fig = go.Figure()

