from calculateHW2P5 import simulate_culprit, simulate_crumbs, combination, print_probabilities, analytic_probabilities, print_analytic_probabilities
import numpy as np

if __name__ == "__main__":
    #N = 100000 #used while testing
    NSize = [1000, 10000, 100000]
    #Priors for each suspect
    pSuperman = 0.5
    pBatman = 0.5

    #Likelihoods of crumbs on each location
    supermanProb = np.array([0.3, 0.7, 0.2])
    batmanProb = np.array([0.4, 0.6, 0.3])

    #Simulate
    '''
    culprit = simulate_culprit(N, pSuperman)
    crumbResults = simulate_crumbs(N, supermanProb, batmanProb, culprit)

    print("Simulation:")
    print_probabilities(combination(crumbResults, culprit), N)
    '''
    print("Simulation:")
    for N in NSize:
        print(f"N = {N}")
        # Simulate culprit and crumbs
        culprit = simulate_culprit(N, pSuperman)
        crumbResults = simulate_crumbs(N, supermanProb, batmanProb, culprit)

        # Count combinations and print probabilities
        print("Simulated Probabilities:")
        print_probabilities(combination(crumbResults, culprit), N)
        print("\n")

    #Analytic
    print("Analytic:")
    print_analytic_probabilities(analytic_probabilities(pSuperman, pBatman, supermanProb, batmanProb))

'''
As the value of N increases, the simulated probabilities get closer to the analytic probabilities. 
'''
