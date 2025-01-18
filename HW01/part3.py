import random

def monteCarlo(numPoints: int) -> float:
    pInside = 0
    
    for _ in range(numPoints):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        
        # (x^2 + y^2 <= 1)
        if x**2 + y**2 <= 1:
            pInside += 1
    
    # approx π
    estimate = 4 * (pInside / numPoints)
    return estimate


# I don't know if an example is needed, but I added one anyways
numPoints = 1000000  # higher -> more accurate
piApprox = monteCarlo(numPoints)
print(f"Approximated value of π: {piApprox}")
