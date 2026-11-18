# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 to maximize sum of radii.
    Uses hexagonal lattice initialization followed by constrained optimization.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Rectangle dimensions - try different aspect ratios to find optimal packing
    best_sum = 0
    best_circles = None
    
    # Test different width/height ratios
    ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    for ratio in ratios:
        width = 2 * ratio / (1 + ratio)
        height = 2 / (1 + ratio)
        
        # Try different initializations and optimization strategies
        circles = _optimize_circles(21, width, height)
        current_sum = np.sum(circles[:, 2])
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = circles.copy()
    
    # Final fallback if no optimization worked
    if best_circles is None:
        best_circles = _fallback_initialization(21, 1.2, 0.8)
    
    return best_circles

def _hexagonal_initialization(n, w, h):
    """Create initial configuration using hexagonal lattice"""
    # For 21 circles, we'll use a 5x5 grid with hexagonal offset
    rows = 5
    cols = 5
    
    positions = []
    initial_radius = min(w, h) * 0.1
    
    # Calculate spacing based on hexagonal packing
    spacing_x = initial_radius * 2.0 * 0.95
    spacing_y = initial_radius * 2.0 * math.sqrt(3) / 2 * 0.95
    
    # Generate hexagonal pattern
    for i in range(rows):
        for j in range(cols):
            if len(positions) < n:
                x_offset = (i % 2) * spacing_x / 2
                x = x_offset + (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                
                # Ensure circle is within bounds
                if (x - initial_radius >= 0 and x + initial_radius <= w and 
                    y - initial_radius >= 0 and y + initial_radius <= h):
                    positions.append([x, y])
    
    # Fill remaining positions if needed
    while len(positions) < n:
        # Add more positions in a structured way
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                if len(positions) >= n:
                    break
                x = j * (w / (cols + 1))
                y = i * (h / (rows + 1))
                positions.append([x, y])
    
    return np.array(positions[:n])

def _optimize_circles(n, width, height):
    """Optimize circle placement using scipy optimization"""
    
    # Get initial positions
    initial_positions = _hexagonal_initialization(n, width, height)
    
    # Define objective function (minimize negative sum of radii)
    def objective(vars):
        positions = vars.reshape(-1, 3)
        radii = positions[:, 2]
        return -np.sum(radii)  # Negative because we minimize
    
    # Define non-overlap constraints
    def non_overlap_constraint(vars):
        positions = vars.reshape(-1, 3)
        n = len(positions)
        constraints = []
        
        # For each pair of circles, ensure distance >= sum of radii
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, r1 = positions[i]
                x2, y2, r2 = positions[j]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                # Constraint: distance >= r1 + r2 (for non-overlap)
                constraints.append(distance - (r1 + r2))
        
        return np.array(constraints)
    
    # Define boundary constraints
    def boundary_constraint(vars):
        positions = vars.reshape(-1, 3)
        constraints = []
        
        # Ensure each circle stays within rectangle bounds
        for i in range(len(positions)):
            x, y, r = positions[i]
            constraints.extend([
                x - r,                    # x >= r
                y - r,                    # y >= r
                width - x - r,            # width - x >= r
                height - y - r            # height - y >= r
            ])
            
        return np.array(constraints)
    
    # Flatten initial configuration for optimization
    initial_vars = np.zeros(n * 3)
    for i in range(n):
        initial_vars[3*i:3*i+2] = initial_positions[i]
        # Set initial radii with more conservative values
        initial_vars[3*i+2] = min(0.06, 0.03 * min(width, height))
    
    # Set up constraints
    non_overlap_cons = {'type': 'ineq', 'fun': non_overlap_constraint}
    boundary_cons = {'type': 'ineq', 'fun': boundary_constraint}
    
    # Set bounds for variables - more careful bounds for radii
    bounds = []
    for i in range(n):
        bounds.extend([(0, width), (0, height), (0.001, min(width, height)/2)])
    
    # Run optimization with carefully tuned parameters
    try:
        # Stage 1: Use L-BFGS-B for coarse optimization
        result1 = minimize(
            objective,
            initial_vars,
            method='L-BFGS-B',
            bounds=bounds,
            constraints=[boundary_cons],
            options={'maxiter': 200, 'ftol': 1e-5}
        )
        
        if result1.success:
            # Stage 2: Fine optimization with SLSQP using all constraints
            result = minimize(
                objective,
                result1.x,
                method='SLSQP',
                bounds=bounds,
                constraints=[non_overlap_cons, boundary_cons],
                options={'maxiter': 200, 'ftol': 1e-7}
            )
            
            if result.success:
                optimal_positions = result.x.reshape(-1, 3)
                # Ensure all radii are valid
                optimal_positions[:, 2] = np.maximum(optimal_positions[:, 2], 0.001)
                return optimal_positions
                
    except Exception:
        pass
    
    # If optimization fails, return initial configuration with slightly adjusted radii
    positions = np.zeros((n, 3))
    for i in range(n):
        x, y = initial_positions[i]
        # Use consistent radius value
        r = min(0.07, 0.04 * min(width, height))
        positions[i] = [x, y, r]
    
    return positions

def _fallback_initialization(n, width, height):
    """Simple fallback initialization when optimization fails"""
    positions = np.zeros((n, 3))
    
    # Use a regular grid pattern
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    x_spacing = width / (cols + 1)
    y_spacing = height / (rows + 1)
    
    radius = min(width, height) * 0.05
    idx = 0
    
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 1) * x_spacing
            y = (i + 1) * y_spacing
            positions[idx] = [x, y, radius]
            idx += 1
    
    return positions
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
