from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d(pose, edges, ax):

    ax.scatter(pose.T[:, 0], pose.T[:, 1], zs=pose.T[:, 2], color='b')

    for u, v in edges:
        ax.plot([pose.T[u, 0], pose.T[v, 0]],
                [pose.T[u, 1], pose.T[v, 1]],
                [pose.T[u, 2], pose.T[v, 2]],
                color='k')