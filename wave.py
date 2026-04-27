def evolve_wave(u, u_prev, dx, dt, c):
    d2u = (np.roll(u,1) - 2*u + np.roll(u,-1)) / dx**2
    u_next = 2*u - u_prev + (c**2 * dt**2) * d2u
    return u_next
