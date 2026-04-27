import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# ==============================
# 1. Setup Parameters
# ==============================
L, Nx = 10.0, 400
dx = L / Nx
x = np.linspace(0, L, Nx)

dt = 0.001
Nt = 500   # number of time steps

k0, sigma = 15.0, 0.5
c = 10.0   # wave speed for Maxwell

# ==============================
# 2. Initial Conditions
# ==============================
psi = np.exp(-(x - 3.0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)

u = np.exp(-(x - 3.0)**2 / (2 * sigma**2)) * np.cos(k0 * x)
u_prev = np.exp(-(x - (3.0 - c*dt))**2 / (2 * sigma**2)) * np.cos(k0 * (x - c*dt))

# ==============================
# 3. Storage for Visualization
# ==============================
psi_data = []
u_data = []

k = 2 * np.pi * np.fft.fftfreq(Nx, dx)

# ==============================
# 4. Time Evolution
# ==============================
for t in range(Nt):

    # --- Schrödinger (Spectral Method) ---
    psi_k = np.fft.fft(psi) * np.exp(-0.5j * (k**2) * dt)
    psi = np.fft.ifft(psi_k)

    psi[0] = psi[-1] = 0

    # --- Maxwell/Wave Equation ---
    d2u = (np.roll(u, 1) - 2*u + np.roll(u, -1)) / dx**2
    u_next = 2*u - u_prev + (c**2 * dt**2) * d2u

    u_prev, u = u, u_next
    u[0] = u[-1] = 0

    # Store data
    psi_data.append(np.abs(psi)**2)
    u_data.append(u.copy())

psi_data = np.array(psi_data)
u_data = np.array(u_data)

# ==============================
# 5. 2D Animation
# ==============================
fig, ax = plt.subplots()
line1, = ax.plot([], [], label="|psi|^2 (Quantum)")
line2, = ax.plot([], [], label="u (Wave)")

ax.set_xlim(0, L)
ax.set_ylim(-1, 2)
ax.legend()

def animate(i):
    line1.set_data(x, psi_data[i])
    line2.set_data(x, u_data[i])
    return line1, line2

ani = animation.FuncAnimation(fig, animate, frames=Nt, interval=20)

plt.title("Schrödinger vs Maxwell Evolution")
plt.show()

# ==============================
# 6. 3D Surface Plot (Quantum)
# ==============================
X, T = np.meshgrid(x, np.arange(Nt))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, T, psi_data, cmap='viridis')

ax.set_xlabel("x")
ax.set_ylabel("time")
ax.set_zlabel("|psi|^2")

plt.title("Quantum Probability Density Evolution")
plt.show()

# ==============================
# 7. 3D Surface Plot (Wave)
# ==============================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, T, u_data, cmap='plasma')

ax.set_xlabel("x")
ax.set_ylabel("time")
ax.set_zlabel("u(x,t)")

plt.title("Classical Wave Evolution")
plt.show()
