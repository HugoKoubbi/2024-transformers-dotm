import numpy as np
import scipy as sp
import imageio
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from datetime import datetime
import networkx as nx
import seaborn as sns

def g(x, beta):
    return -np.sin(x) * np.exp(beta * np.cos(x))

def h(x, beta):
    return (np.cos(x) - beta * np.sin(x) ** 2) * np.exp(beta * np.cos(x))

def energy(theta, beta):
    E = 0
    n = len(theta)
    for i in range(n):
        E += np.sum(np.exp(beta * np.cos(theta[i] * np.ones_like(theta) - theta)))
    return E / (np.exp(beta) * n ** 2)

def energy_softmax(theta, beta):
    E = 0
    n = len(theta)
    for i in range(n):
        E += np.sum(np.exp(beta * (np.cos(theta[i] * np.ones_like(theta) - theta) - 1)))
    return E / n ** 2

def grad(theta, beta):
    g = np.zeros_like(theta)
    n = len(theta)
    for i in range(n):
        g[i] = 1 / n ** 2 * np.sum(-np.sin(theta[i] * np.ones_like(theta) - theta) * np.exp(beta * np.cos(theta[i] * np.ones_like(theta) - theta)))
    return n * g

def hess(theta, beta):
    n = len(theta)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                H[i, j] = g(theta[i] - theta[j], beta)
        H[i, i] = -np.sum(H[i, :])
    return H / n ** 2

def strict_saddle_point(theta0, beta):
    theta = theta0
    Hessian = hess(theta, beta)
    eigenvalues, _ = np.linalg.eig(Hessian)
    epsilon = 10e-18
    if any(eig > epsilon for eig in eigenvalues) and any(eig < -epsilon for eig in eigenvalues):
        return True, eigenvalues
    return False, eigenvalues

def find_saddle_point(theta0, beta, lr, saddle_crit, max_iter):
    theta = theta0
    L_saddle = []
    for i in range(max_iter):
        grad = g(theta, beta)
        theta = theta + lr * grad
        if np.linalg.norm(grad) < saddle_crit:
            if strict_saddle_point(theta, beta)[0]:
                L_saddle.append((theta, strict_saddle_point(theta, beta)[1]))
    return L_saddle

def gradient_descent(theta0, lr, beta, tol, max_iter):
    theta = theta0
    for i in range(max_iter):
        grad = g(theta, beta)
        theta = theta + lr * grad
        if np.linalg.norm(grad) < tol:
            break
    return theta

def gradient_descent_check(theta0, lr, beta, tol, max_iter):
    theta = theta0
    E = []
    for i in range(max_iter):
        gradient = grad(theta, beta)
        theta = theta + lr * gradient
        E.append(energy(theta, beta))
        if np.linalg.norm(gradient) < tol and 1 == 0:
            break
    return theta, E

def gradient_descent_norm(theta0, lr, beta, tol, max_iter):
    theta = theta0
    gradient_norm = []
    for i in range(max_iter):
        gradient = grad(theta, beta)
        theta = theta + lr * gradient
        gradient_norm.append(np.linalg.norm(gradient))
        if np.linalg.norm(gradient) < tol:
            break
    return theta, gradient_norm

def proj(y, x):
    return y - np.dot(x, y) * x

def grad_softmax(theta, beta):  
    g = np.zeros_like(theta)
    n = len(theta)
    for i in range(n):
        Z_i = np.sum(np.exp(beta * np.dot(theta[i], theta)))
        g[i] = proj(1 / Z_i * np.sum(np.exp(beta * np.dot(theta[i], theta)) * theta), theta[i])
    return g

def grad_soft(theta, beta):  
    g = np.zeros_like(theta)
    n = len(theta)
    for i in range(n):
        Z_i = np.sum(np.exp(beta * (np.cos(theta[i] * np.ones_like(theta) - theta) - 1)))
        g[i] = (1 / Z_i) * np.sum(-np.sin(theta[i] * np.ones_like(theta) - theta) * np.exp(beta * (np.cos(theta[i] * np.ones_like(theta) - theta) - 1)))
    return g

def grad_USA(theta, beta):  
    g = np.zeros_like(theta)
    n = len(theta)
    for i in range(n):
        g[i] = (1 / n ** 2) * np.sum(-np.sin(theta[i] * np.ones_like(theta) - theta) * np.exp(beta * (np.cos(theta[i] * np.ones_like(theta) - theta) - 1)))
    return g

def energy_softmax(theta, beta):
    E = 0
    n = len(theta)
    for i in range(n):
        E += np.sum(np.exp(beta * (np.cos(theta[i] * np.ones_like(theta) - theta) - 1)))
    return E / n ** 2

def gradient_descent_softmax(theta0, lr, beta, tol, max_iter):
    theta = theta0
    E = []
    for i in range(max_iter):
        gradient = grad_soft(theta, beta)
        theta = theta + lr * gradient
        E.append(energy(theta, beta))
    return theta, E

import numpy as np
import matplotlib.pyplot as plt

def algo_traj_1(theta0):
    theta = theta0
    M = len(theta0)
    theta_weight = np.ones_like(theta)
    L_collision = []
    L_time = []
    L_energy = []
    L_dist_mem = []
    n = len(theta)
    
    while n >= 2:
        L_dist = []
        L_dir = []
        
        for i in range(n):
            j_i = np.argmin([np.abs((theta[i] - theta[j]) % (2 * np.pi)) for j in range(n) if j != i])
            j_i = j_i if j_i < i else j_i + 1
            L_dir.append(j_i)
            L_dist.append(np.abs((theta[i] - theta[j_i]) % (2 * np.pi)))
        
        # Calculating collision
        col = np.argmin(L_dist)
        j_i = L_dir[col]
        L_collision.append((col, j_i))
        L_time.append(1 - np.cos((theta[col] - theta[j_i]) % (2 * np.pi)))
        L_energy.append(2 * (theta_weight[col] * theta_weight[j_i]) / M ** 2)
        L_dist_mem.append(np.abs((theta[col] - theta[j_i]) % (2 * np.pi)))

        # Updating theta and theta_weight
        theta[j_i] = ((theta[j_i] * theta_weight[j_i] + theta[col] * theta_weight[col]) / (theta_weight[j_i] + theta_weight[col])) % (2 * np.pi)
        theta_weight[j_i] += theta_weight[col]
        theta = np.delete(theta, col)
        theta_weight = np.delete(theta_weight, col)
        n = len(theta)

    return L_collision, L_time, L_dir, L_dist_mem, L_energy

def piecewise_constant_function(L_t, L_ene, x):
    y = np.zeros_like(x)
    for i in range(len(L_t) - 1):
        y[(x >= L_t[i]) & (x < L_t[i+1])] = L_ene[i]
    y[x >= L_t[-1]] = L_ene[-1]
    return y

def algo_traj_2(theta0):
    L_collision, L_time, L_dir, L_dist_mem, L_energy = algo_traj_1(theta0)
    n = len(theta0)
    L_ene = [np.sum(L_energy[:i]) + 1 / n for i in range(len(L_energy) + 1)]
    L_t = [np.sum(L_time[:i]) for i in range(len(L_time) + 1)]
    x = np.linspace(0, 0.5 + L_t[-1], 1000)
    y_values = piecewise_constant_function(L_t, L_ene, x)
    
    plt.figure(1)
    plt.plot(L_t, L_ene, 'ro')
    plt.plot(x, y_values)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Energy as a function of time')
    plt.show()
    
    return L_collision, L_time, L_dir, L_dist_mem, L_energy

def energy_weight(theta, theta_weight, beta):
    E = 0
    n = len(theta)
    for i in range(n):
        for j in range(n):
            E += np.exp(beta * (np.cos(theta[i] - theta[j]) - 1)) * theta_weight[i] * theta_weight[j]
    return E

def piecewise_constant_function(jump_points, jump_values, x_values):
    y_values = np.zeros_like(x_values)
    for point, value in zip(jump_points, jump_values):
        y_values[x_values >= point] = value
    return y_values

def algo_traj_3(theta0, lr, beta, max_iter):
    theta = theta0
    M = len(theta0)
    theta_weight = np.ones_like(theta) 
    E = []
    n = len(theta)
    c = 0
    while n >= 2 and (c < max_iter):
        c += 1
        L_dist = []
        L_nn = []
        g = np.zeros_like(theta)
        for i in range(n):
            j_i = np.argmin([np.min(((theta[i] - theta[j]) % (2 * np.pi), 2 * np.pi - (theta[i] - theta[j]) % (2 * np.pi))) for j in range(n) if j != i])
            j_i = j_i if j_i < i else j_i + 1
            L_dist.append(np.min([np.min(((theta[i] - theta[j]) % (2 * np.pi), 2 * np.pi - (theta[i] - theta[j]) % (2 * np.pi))) for j in range(n) if j != i]))
            L_nn.append(j_i)
            Z_i = np.sum(np.exp(beta * (np.cos(theta[i] * np.ones_like(theta) - theta) - 1)))
            g[i] = 1 / Z_i * (-(theta_weight[i] + theta_weight[j_i]) * np.sin(theta[i] - theta[j_i]) * np.exp(beta * (np.cos(theta[i] - theta[j_i]) - 1)))
        
        # Updating theta 
        theta = theta + lr * g
        E.append(energy_weight(theta, theta_weight, beta) / M ** 2)
        for i in range(n):
            if L_dist[i] < 1 / beta ** 2:
                j_i = L_nn[i]
                L_dist[j_i] = 1
                theta_weight[j_i] = theta_weight[j_i] + theta_weight[i]
                theta = np.delete(theta, i)
                theta_weight = np.delete(theta_weight, i)
        n = len(theta)
    return theta, E

def algo_traj_4(theta0, lr, beta, max_iter):
    theta = theta0
    M = len(theta0)
    theta_weight = np.ones_like(theta0) 
    E = []
    n = len(theta)
    c = 0
    while n >= 2 and (c < max_iter):
        c += 1
        L_dist = []
        L_nn = []
        g = np.zeros_like(theta)
        for i in range(n):
            j_i = np.argmax([np.cos(theta[i] - theta[j]) for j in range(n) if j != i])
            L_dist.append(np.max([np.cos(theta[i] - theta[j]) for j in range(n) if j != i]))
            j_i = j_i if j_i < i else j_i + 1
            L_nn.append(j_i)
            Z_i = np.sum(np.exp(beta * (np.cos(theta[i] * np.ones_like(theta) - theta) - 1)))
            g[i] = 1 / Z_i * (-(theta_weight[i] + theta_weight[j_i]) * np.sin(theta[i] - theta[j_i]) * np.exp(beta * (np.cos(theta[i] - theta[j_i]) - 1)))
        
        # Updating theta 
        theta = (theta + lr * g) % (2 * np.pi)
        E.append(energy_weight(theta, theta_weight, beta) / M ** 2)
        
        # Suppression des particules qui sont dans une fenêtre d'interaction proche
        for i in range(n):
            if L_dist[i] > 1 - 1 / (2 * beta):
                j = L_nn[i]
                L_dist[j] = 0
                theta_weight[j] = theta_weight[j] + theta_weight[i]
                theta = np.delete(theta, i)
        n = len(theta)
    return theta, E


def tau_lim(theta0, lr, beta, max_iter):
    theta = theta0
    M = len(theta0)
    theta_weight = np.ones_like(theta0)
    E = []
    n = len(theta)
    c = 0
    while n >= 2 and (c < max_iter):
        c += 1
        L_dist = []
        L_nn = []
        g = np.zeros_like(theta)
        for i in range(n):
            j_i = np.argmax([np.cos(theta[i] - theta[j]) for j in range(n) if j != i])
            L_dist.append(np.max([np.cos(theta[i] - theta[j]) for j in range(n) if j != i]))
            j_i = j_i if j_i < i else j_i + 1
            L_nn.append(j_i)
        C = np.max(L_dist)
        for i in range(n):
            Z_i = np.sum(np.exp(beta * (-np.cos(theta[i] * np.ones_like(theta) - theta) + C)))
            g[i] = 1 / Z_i * np.sum(-np.sin(theta[i] * np.ones_like(theta) - theta) * np.exp(beta * (-np.cos(theta[i] * np.ones_like(theta) - theta) + C)))
        
        # Updating theta 
        theta = (theta + lr * g) % (2 * np.pi)
        E.append(energy_weight(theta, theta_weight, beta) / M ** 2)
        
        # Suppression des particules qui sont dans une fenêtre d'interaction proche
        for i in range(n):
            if L_dist[i] > 1 - 1 / (2 * beta):
                j = L_nn[i]
                L_dist[j] = 0
                theta_weight[j] = theta_weight[j] + theta_weight[i]
                theta = np.delete(theta, i)
        n = len(theta)
    return theta, E

def tau_lim2(theta0, lr, beta, max_iter):
    theta = theta0
    E = []
    n = len(theta)
    for c in range(max_iter):
        L_dist = []
        g = np.zeros_like(theta)
        for i in range(n):
            S_i = [j for j in range(len(theta)) if np.cos(theta[i] - theta[j]) < 1 - np.log(beta) / beta]
            j_i = np.argmax([np.cos(theta[i] - theta[j]) - 1 for j in S_i])
            L_dist.append(np.cos(theta[i] - theta[j_i]))
        C = np.max(L_dist)
        for i in range(n):
            Z_i = np.sum(np.exp(beta * (np.cos(theta[i] * np.ones_like(theta) - theta) - 1)))
            g[i] = beta / Z_i * np.sum(-np.sin(theta[i] * np.ones_like(theta) - theta) * np.exp(beta * (np.cos(theta[i] * np.ones_like(theta) - theta) - C)))
        # Updating theta 
        theta = (theta + lr * g) % (2 * np.pi)
        E.append(energy_softmax(theta, beta))
    return theta, E

def time_rescale_3(theta0, lr, beta, max_iter):
    theta = theta0
    E = []
    lrad = []
    for i in range(max_iter):
        gradient = grad_soft(theta, beta)
        gradient_norm = np.linalg.norm(gradient)
        theta = theta + lr * beta * gradient / gradient_norm
        lrad.append(lr * gradient_norm)
        E.append(energy(theta, beta))
    return [], E

def time_rescale_6(theta0, lr, beta, max_iter):
    theta = theta0
    thet = []
    E = []
    for i in range(max_iter):
        gradient = grad_soft(theta, beta)
        gradient_norm = np.linalg.norm(gradient)
        theta = theta + lr * gradient / gradient_norm
        if len(E) > 10 and E[-1] - E[-10] > 1 / len(theta) ** 2:
            for j in range(len(theta)):
                for k in range(len(theta)):
                    if k != j and np.abs(theta[j] - theta[k]) < 1 / beta:
                        theta[k] = theta[j]
        E.append(energy(theta, beta))
        thet.append(theta)
    return thet, E

def time_rescale_3(theta0, lr, beta, max_iter):
    theta = theta0
    thet = []
    E = []
    for i in range(max_iter):
        gradient = grad_USA(theta, beta)
        gradient_norm = np.linalg.norm(gradient)
        theta = theta + lr * np.log(beta) * gradient / gradient_norm
        if gradient_norm < 10e-6:
            for j in range(len(theta)):
                for k in range(j + 1, len(theta)):
                    if np.abs(theta[j] - theta[k]) < 10e-4:
                        theta[k] = theta[j]
        E.append(energy_softmax(theta, beta))
    return thet, E

def time_rescale_4(theta0, lr, beta, max_iter):
    theta = theta0
    thet = []
    E = []
    for i in range(max_iter):
        gradient = grad_soft(theta, beta)
        gradient_norm = np.linalg.norm(gradient)
        theta = theta + lr * gradient / gradient_norm ** (3 / 2)
        E.append(energy(theta, beta))
        thet.append(theta)
    return thet, E


np.random.seed(200)

mathematica_colors = {
    'red': (0.8, 0.2, 0.2),
    'blue': (0.2, 0.2, 0.8),
    'orange': (0.8, 0.4, 0.0),
    'green': (0.2, 0.6, 0.2),
    'magenta': (0.8, 0.2, 0.8),
    'yellow': (0.8, 0.8, 0.0)
}


color = [mathematica_colors['magenta'],
         mathematica_colors['blue'], 
         mathematica_colors['green'], 
         mathematica_colors['yellow'],
         mathematica_colors['orange'], 
         mathematica_colors['red']]

n = 5
#n = 80
beta = 2000

# theta0 = np.array([
#     0,
#     (1 * np.log(beta) / beta ** (1 / 2)) % np.pi,
#     (3 * np.log(beta) / beta ** (1 / 2)) % np.pi,
#     (7 * np.log(beta) / beta ** (1 / 2)) % np.pi,
#     (10 * np.log(beta) / beta ** (1 / 2)) % np.pi
# ])

theta0 = 0.5 * np.array([
    0,
    (0.5 * np.log(beta) / beta**(1/2)) % np.pi,
    (2 * np.log(beta) / beta**(1/2)) % np.pi,
    (4 * np.log(beta) / beta**(1/2)) % np.pi,
    (7 * np.log(beta) / beta**(1/2)) % np.pi
])

print(theta0)

lr = 10e-6
max_iter = int(1e6)
fig, ax = plt.subplots()

x_coords = np.cos(theta0)
y_coords = np.sin(theta0)

# Ensure the grid is drawn below the circle and points
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Draw the grid
ax.grid(color='lightgray', linestyle=(0, (1, 10)))

# Plot the circle and points
circle = plt.Circle((0, 0), 1, color=mathematica_colors['blue'], fill=False)
ax.add_artist(circle)
ax.set_aspect('equal', adjustable='box')
ax.plot(x_coords, y_coords, 'o', markerfacecolor='none', markeredgecolor=mathematica_colors['red'])

plt.savefig("circle.pdf", format='pdf', bbox_inches='tight')

from tqdm import tqdm

def time_rescale_5(theta0, lr, beta, max_iter):
    theta = theta0
    thet = []
    E = []
    G_d = []

    for i in tqdm(range(max_iter), desc=f"Processing beta={beta}"):
        gradient = grad_soft(theta, beta)
        gradient_norm = np.linalg.norm(gradient)
        theta = theta + lr * np.log(beta) * gradient / gradient_norm
        E.append(energy(theta, beta))
        G_d.append(gradient_norm)
    return thet, E

plt.figure()

# Draw the grid
ax.grid(color='lightgray', linestyle=(0, (1, 10)))

# Loop through the beta values and plot on the same figure
for i, beta in enumerate(tqdm([500, 1000, 1500, 2000, 3000, 4000], desc="Beta values")):
    _, E = time_rescale_3(theta0, lr, beta, max_iter)
    print(E)
    L = [lr * t for t in range(len(E))]
    plt.plot(L, E, ls='-', color=color[i])

# Set axis properties and labels
plt.xlabel(r'Rescaled time')
plt.ylabel(r'Energy')
plt.grid(color='lightgray', linestyle=(0, (1, 10)))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlim(0, None)

# Save the figure as a PDF
plt.savefig("time-rescale.pdf", format='pdf', bbox_inches='tight')

