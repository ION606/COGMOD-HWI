import numpy as np

# SIMULATION CALCULATIONS
def simulate_culprit(N, pSuperman):
    return np.random.rand(N) < pSuperman

def simulate_crumbs(N, supermanProb, batmanProb, culprit):
    randomDraw = np.random.rand(N, 3)
    supermanCrumbs = (randomDraw < supermanProb)
    batmanCrumbs = (randomDraw < batmanProb)
    return np.where(culprit[:, None], supermanCrumbs, batmanCrumbs)

def combination(crumbResults, culprit):
    combinations = {}
    for couch in [False, True]:
        for kitchen in [False, True]:
            for gym in [False, True]:
                for culprit_label, culprit_val in [("Superman", True), ("Batman", False)]:
                    mask = (crumbResults[:, 0] == couch) & (crumbResults[:, 1] == kitchen) & (crumbResults[:, 2] == gym) & (culprit == culprit_val)
                    combinations[(couch, kitchen, gym, culprit_label)] = np.sum(mask)
    return combinations

def print_probabilities(combinations, N):
    print("Couch:\t Kitchen:  Gym:\t  Culprit:")
    for k, v in combinations.items():
        couch, kitchen, gym, culprit_label = k
        #formatting
        couch_str = 'True ' if couch else 'False'
        kitchen_str = 'True ' if kitchen else 'False'
        gym_str = 'True ' if gym else 'False'
        if culprit_label == 'Batman': culprit_label = 'Batman  '

        print(f"{couch_str}\t {kitchen_str}\t   {gym_str}  {culprit_label}: {(v / N) * 100:.2f}%")

# ANALYTIC CALCULATIONS
def analytic_probabilities(pSuperman, pBatman, supermanProb, batmanProb):
    combinations = {}
    for couch in [False, True]:
        for kitchen in [False, True]:
            for gym in [False, True]:
                #Superman
                prob_superman = pSuperman
                prob_superman *= supermanProb[0] if couch else (1 - supermanProb[0])
                prob_superman *= supermanProb[1] if kitchen else (1 - supermanProb[1])
                prob_superman *= supermanProb[2] if gym else (1 - supermanProb[2])
                combinations[(couch, kitchen, gym, "Superman")] = prob_superman

                #Batman
                prob_batman = pBatman
                prob_batman *= batmanProb[0] if couch else (1 - batmanProb[0])
                prob_batman *= batmanProb[1] if kitchen else (1 - batmanProb[1])
                prob_batman *= batmanProb[2] if gym else (1 - batmanProb[2])
                combinations[(couch, kitchen, gym, "Batman")] = prob_batman

    return combinations

def print_analytic_probabilities(combinations):
    print("Couch:\t Kitchen:  Gym:\t  Culprit:")
    for k, v in combinations.items():
        couch, kitchen, gym, culprit_label = k

        #formatting
        couch_str = 'True ' if couch else 'False'
        kitchen_str = 'True ' if kitchen else 'False'
        gym_str = 'True ' if gym else 'False'
        if culprit_label == 'Batman': culprit_label = 'Batman  '
        print(f"{couch_str}\t {kitchen_str}\t   {gym_str}  {culprit_label}: {v * 100:.2f}%")