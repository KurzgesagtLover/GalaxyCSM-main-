import sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# Ensure we can import from local gce package
sys.path.append(os.getcwd())
try:
    from gce.stellar import generate_galaxy, stellar_evolution
except ImportError:
    # If run from scripts/ or similar, try modifying path
    sys.path.append(os.path.dirname(os.getcwd()))
    from gce.stellar import generate_galaxy, stellar_evolution

def generate_background():
    print("Generating synthetic Gaia-like stellar population...")
    # Generate large sample to get smooth density
    # t_max=20 ensures we have a mix of old and young stars consistent with the model
    data = generate_galaxy(n_stars=150000, params={'t_max': 20.0})
    stars = data['stars']
    
    T_eff = []
    L = []
    
    # Process evolution at current time (13.8 Gyr)
    print("Calculating stellar properties...")
    for i, m in enumerate(stars['mass']):
        age = max(13.8 - stars['birth'][i], 0)
        birth_z = stars.get('birth_z', [0.02] * len(stars['mass']))[i]
        evo = stellar_evolution(m, age, metallicity_z=birth_z)
        t, l = evo['T_eff'], evo['luminosity']
        
        # Filter visibility (Gaia limit approx)
        if t < 100 or l < 1e-6: continue
        
        T_eff.append(t)
        L.append(l)
        
        # Simulate Binaries (Gaia sees many)
        # Adds a "second sequence" slightly above MS
        if np.random.random() < 0.35:
            # Companion
            L.append(l * (1 + np.random.uniform(0.2, 1.0)))
            T_eff.append(t * np.random.uniform(0.96, 1.0)) 

    # Observational scatter (simulate measurement error)
    T_eff = np.array(T_eff) * np.random.normal(1, 0.015, len(T_eff)) 
    L = np.array(L) * np.random.normal(1, 0.08, len(L))
    
    print(f"Plotting {len(T_eff)} sources...")
    
    # Setup plot - matches app.js ranges
    # X: 5.2 -> 3.2 (reversed)
    # Y: -5 -> 7
    fig = plt.figure(figsize=(6, 9), facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('black')
    
    # 2D Histogram (Density Map)
    # range=[[3.2, 5.2], [-5, 7]] corresponds to X from 3.2 to 5.2.
    # We will invert X axis later, so 5.2 is on Left.
    h = ax.hist2d(np.log10(T_eff), np.log10(L), bins=[400, 600], 
                  range=[[3.2, 5.2], [-5, 7]],
                  cmap='inferno', norm=LogNorm(vmin=1, vmax=len(T_eff)/150))
    
    ax.set_xlim(5.2, 3.2) # Hot (left) to Cool (right)
    ax.set_ylim(-5, 7)    # Dim (bottom) to Bright (top)
    ax.axis('off')
    
    os.makedirs('static/img', exist_ok=True)
    out_path = 'static/img/hr_background.png'
    plt.savefig(out_path, dpi=150, bbox_inches=None, pad_inches=0)
    print(f"Saved {out_path}")

if __name__ == '__main__':
    generate_background()
