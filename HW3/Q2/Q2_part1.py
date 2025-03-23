import numpy as np
import matplotlib.pyplot as plt


def simulate_ddm(v, a=1.0, beta=0.5, tau=0.3, sigma=1.0, dt=0.001, max_steps=3000):
    X = beta * a  # start position
    t = 0.0
    for _ in range(max_steps):
        dW = np.random.normal(0, np.sqrt(dt))
        dX = v * dt + sigma * dW
        X += dX
        t += dt
        if X >= a:
            return t + tau, 1  # upper bound hit
        elif X <= 0:
            return t + tau, 0  # lower bound hit
    return max_steps * dt + tau, None  # Timeout (optional)


# terrible params (upped in part 2)
vs = np.linspace(0.5, 1.5, 25)  # drift rates for test
n_trials = 2000

# store
upper_means, lower_means = [], []

for v in vs:
    upper_rts, lower_rts = [], []
    for _ in range(n_trials):
        rt, choice = simulate_ddm(v)
        if choice == 1:
            upper_rts.append(rt)
        elif choice == 0:
            lower_rts.append(rt)
    # means (ignore cases where no hits)
    upper_means.append(np.mean(upper_rts) if upper_rts else np.nan)
    lower_means.append(np.mean(lower_rts) if lower_rts else np.nan)

# plotting yay
plt.figure(figsize=(10, 6))
plt.plot(vs, upper_means, 'o-', label='Upper Boundary Mean RT')
plt.plot(vs, lower_means, 's-', label='Lower Boundary Mean RT')
plt.plot(vs, np.array(upper_means) - np.array(lower_means),
         'd-', label='Mean Difference')
plt.xlabel('Drift Rate (v)')
plt.ylabel('Response Time (s)')
plt.title('Effect of Drift Rate on RT Distributions')
plt.legend()
plt.grid(True)
plt.savefig('part1.png')
