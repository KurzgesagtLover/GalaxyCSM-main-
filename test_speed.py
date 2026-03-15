import time
import numpy as np
from gce.planets import generate_planets

def test_speed():
    rng = np.random.default_rng(42)
    masses = rng.uniform(0.1, 2.0, 25000)
    mets = rng.uniform(-1.0, 0.5, 25000)
    
    t0 = time.time()
    count_planets = 0
    count_hz = 0
    for i in range(25000):
        planets = generate_planets(masses[i], mets[i], rng)
        if len(planets) > 0:
            count_planets += 1
            if any(p.get('in_hz', False) for p in planets):
                count_hz += 1
                
    t1 = time.time()
    print(f"Generated planets for 25000 stars in {t1-t0:.3f} s")
    print(f"Has planets: {count_planets}, Has HZ: {count_hz}")

if __name__ == '__main__':
    test_speed()
