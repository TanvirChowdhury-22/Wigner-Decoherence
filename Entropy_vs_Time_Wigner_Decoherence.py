{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww35800\viewh22520\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import numpy as np\
import matplotlib.pyplot as plt\
from scipy.integrate import odeint\
\
gamma = 0.3        \
Omega = 1.0        \
T = 0.0            \
Lambda = 1000.0    # Cutoff frequency\
kB = 1.0          \
hbar = 1.0         \
\
# Initial conditions\
r = np.log(8)\
sigma_q0 = 0.5 * np.exp(2 * r)\
sigma_p0 = 0.5 * np.exp(-2 * r)\
sigma_qp0 = 0.0 \
\
# Initial state vector\
y0 = [sigma_q0, sigma_p0, sigma_qp0]\
\
\
t = np.linspace(0, 20, 500)\
\
\
def derivatives(y, t, gamma, Omega):\
    sigma_q, sigma_p, sigma_qp = y\
    dsigma_q_dt = 2 * sigma_qp\
    dsigma_p_dt = -2 * Omega**2 * sigma_qp - 4 * gamma * sigma_p + 2 * gamma * hbar\
    dsigma_qp_dt = sigma_p - Omega**2 * sigma_q - 2 * gamma * sigma_qp\
    return [dsigma_q_dt, dsigma_p_dt, dsigma_qp_dt]\
\
\
sol = odeint(derivatives, y0, t, args=(gamma, Omega))\
\
\
sigma_q = sol[:, 0]\
sigma_p = sol[:, 1]\
sigma_qp = sol[:, 2]\
\
\
def compute_entropy(sigma_q, sigma_p, sigma_qp):\
    det_sigma = sigma_q * sigma_p - sigma_qp**2\
    nu = np.sqrt(det_sigma)\
    S = (nu + 0.5) * np.log(nu + 0.5) - (nu - 0.5) * np.log(nu - 0.5)\
    return S\
\
\
entropies = compute_entropy(sigma_q, sigma_p, sigma_qp)\
\
\
plt.plot(t, entropies)\
plt.xlabel("Time")\
plt.ylabel("Entropy")\
plt.title("Entropy vs Time")\
plt.grid(True)\
plt.tight_layout()\
plt.savefig("entropy_vs_time.png", dpi=300)\
plt.show()}