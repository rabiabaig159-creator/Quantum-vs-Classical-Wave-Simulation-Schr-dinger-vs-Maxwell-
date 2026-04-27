def evolve_schrodinger(psi, k, dt):
    psi_k = np.fft.fft(psi)
    psi_k *= np.exp(-0.5j * (k**2) * dt)
    return np.fft.ifft(psi_k)
