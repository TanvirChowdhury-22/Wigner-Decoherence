{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import numpy as np\
import matplotlib.pyplot as plt\
from scipy.integrate import odeint\
import matplotlib.animation as animation\
from matplotlib.patches import Ellipse\
\
\
gamma = 0.3\
Omega = 1.0\
T = 1.0\
hbar = 1.0\
kB = 1.0\
m = 1.0\
\
# Initial conditions\
r = np.log(8)\
sigma_q0 = 0.5 * np.exp(2 * r)\
sigma_p0 = 0.5 * np.exp(-2 * r)\
sigma_qp0 = 0.0\
Q0 = 0.0\
P0 = 10.0\
\
# state vector\
y0 = [sigma_q0, sigma_p0, sigma_qp0, Q0, P0]\
t = np.linspace(0, 20, 400)\
\
def derivatives_extended(y, t, gamma, Omega, T, hbar, kB):\
    sigma_q, sigma_p, sigma_qp, Q, P = y\
    d_sigma_q_dt = 2 * sigma_qp\
    d_sigma_p_dt = -2 * Omega**2 * sigma_qp - 4 * gamma * sigma_p + 4 * gamma * kB * T\
    d_sigma_qp_dt = sigma_p - Omega**2 * sigma_q - 2 * gamma * sigma_qp\
    d_Q_dt = P / m\
    d_P_dt = -Omega**2 * Q - 2 * gamma * P\
    return [d_sigma_q_dt, d_sigma_p_dt, d_sigma_qp_dt, d_Q_dt, d_P_dt]\
\
sol = odeint(derivatives_extended, y0, t, args=(gamma, Omega, T, hbar, kB))\
sigma_q, sigma_p, sigma_qp, Q_mean, P_mean = sol.T\
\
def compute_entropy(sigma_q, sigma_p, sigma_qp):\
    det_sigma = sigma_q * sigma_p - sigma_qp**2\
    nu = np.sqrt(det_sigma)\
    with np.errstate(divide='ignore', invalid='ignore'):\
        S = (nu + 0.5) * np.log(nu + 0.5) - (nu - 0.5) * np.log(nu - 0.5)\
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)\
    return S\
\
entropies = compute_entropy(sigma_q, sigma_p, sigma_qp)\
\
# Plot entropy\
plt.figure(figsize=(6, 4))\
plt.plot(t, entropies)\
plt.xlabel("Time")\
plt.ylabel("Entropy S")\
plt.title("Entropy vs Time (Finite Temperature, Displaced State)")\
plt.grid(True)\
plt.tight_layout()\
plt.savefig("entropy_vs_time_temp.png", dpi=300)\
plt.show()\
\
fig, ax = plt.subplots(figsize=(5, 5))\
ax.set_xlim(-20, 20)\
ax.set_ylim(-20, 20)\
ax.set_xlabel("Q")\
ax.set_ylabel("P")\
ax.set_title("Wigner Ellipse Evolution")\
\
ellipse = Ellipse(xy=(0, 0), width=1, height=1, angle=0, edgecolor='r', fc='None', lw=2)\
ax.add_patch(ellipse)\
mean_dot, = ax.plot([], [], 'bo')\
\
def init():\
    ellipse.set_width(0.1)\
    ellipse.set_height(0.1)\
    ellipse.set_angle(0)\
    mean_dot.set_data([], [])\
    return ellipse, mean_dot\
\
def update(frame):\
    cov = np.array([[sigma_q[frame], sigma_qp[frame]],\
                    [sigma_qp[frame], sigma_p[frame]]])\
    vals, vecs = np.linalg.eigh(cov)\
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))\
    width, height = 2 * np.sqrt(vals)\
    ellipse.set_width(width)\
    ellipse.set_height(height)\
    ellipse.set_angle(angle)\
    ellipse.set_center((Q_mean[frame], P_mean[frame]))\
    mean_dot.set_data([Q_mean[frame]], [P_mean[frame]])\
    return ellipse, mean_dot\
\
ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init,\
                              blit=True, interval=50)\
\
ani.save("wigner_ellipse_evolution.gif", writer='pillow', fps=30)}