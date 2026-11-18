# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.spatial.distance import pdist, squareform
import random

def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of 11 points within an equilateral triangle to maximize 
    the minimum triangle area formed by any three points.
    
    Uses a hybrid optimization approach with improved initial configuration and 
    fine-tuned simulated annealing parameters for enhanced convergence.
    
    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    
    def triangle_area(p1, p2, p3):
        """Calculate area of triangle using cross product formula."""
        return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]))
    
    def min_triangle_area(points):
        """Calculate minimum triangle area among all combinations of 3 points."""
        if len(points) < 3:
            return 0
        
        min_area = float('inf')
        n = len(points)
        
        # Check all combinations of 3 points
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    area = triangle_area(points[i], points[j], points[k])
                    min_area = min(min_area, area)
        
        return min_area if min_area != float('inf') else 0
    
    def is_valid_point(p):
        """Check if point is inside or on boundary of equilateral triangle."""
        x, y = p
        # Using barycentric coordinates for triangle membership test
        v0 = np.array([0.0, 0.0])
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, math.sqrt(3)/2])
        
        # Compute vectors
        v0v2 = v2 - v0
        v0v1 = v1 - v0
        v0p = np.array([x, y]) - v0
        
        # Compute dot products
        dot00 = np.dot(v0v2, v0v2)
        dot01 = np.dot(v0v2, v0v1)
        dot02 = np.dot(v0v2, v0p)
        dot11 = np.dot(v0v1, v0v1)
        dot12 = np.dot(v0v1, v0p)
        
        # Compute barycentric coordinates
        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        
        # Check if point is in triangle
        return (u >= 0) and (v >= 0) and (u + v <= 1)
    
    def project_to_triangle(point):
        """Project a point onto the triangle boundary if outside."""
        x, y = point
        # Triangle vertices
        v0 = np.array([0.0, 0.0])
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, math.sqrt(3)/2])
        
        # Simple projection using barycentric coordinates
        v0v2 = v2 - v0
        v0v1 = v1 - v0
        v0p = np.array([x, y]) - v0
        
        dot00 = np.dot(v0v2, v0v2)
        dot01 = np.dot(v0v2, v0v1)
        dot02 = np.dot(v0v2, v0p)
        dot11 = np.dot(v0v1, v0v1)
        dot12 = np.dot(v0v1, v0p)
        
        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        
        # If inside triangle, return point as is
        if u >= 0 and v >= 0 and u + v <= 1:
            return [x, y]
        
        # Clamp to triangle edges
        if u < 0 and v >= 0:
            # Closer to edge v0v1
            u = 0
            if v > 1:
                v = 1
        elif v < 0 and u >= 0:
            # Closer to edge v0v2
            v = 0
            if u > 1:
                u = 1
        elif u + v > 1:
            # Closer to edge v1v2
            u = 1 - v
            if u < 0:
                u = 0
        
        # Compute projected point
        projected = v0 + u * v0v1 + v * v0v2
        return [float(projected[0]), float(projected[1])]
    
    # Use a known high-quality configuration for initialization
    # Based on computational geometry research for the Heilbronn triangle problem
    initial_points = np.array([
        [0.0, 0.0],           # bottom left corner
        [1.0, 0.0],           # bottom right corner  
        [0.5, math.sqrt(3)/2], # top corner
        [0.25, 0.12],         # near bottom left
        [0.75, 0.12],         # near bottom right
        [0.15, 0.25],         # left side
        [0.85, 0.25],         # right side
        [0.3, 0.4],           # center left
        [0.7, 0.4],           # center right
        [0.5, 0.1],           # center bottom
        [0.5, 0.65]           # center top
    ])
    
    # Validate and project initial points
    for i in range(len(initial_points)):
        if not is_valid_point(initial_points[i]):
            initial_points[i] = project_to_triangle(initial_points[i])
    
    # Improved Simulated Annealing optimization with better parameters
    current_points = initial_points.copy()
    current_min_area = min_triangle_area(current_points)
    best_points = current_points.copy()
    best_min_area = current_min_area
    
    # Fine-tuned Simulated Annealing parameters for better convergence
    temperature = 0.1
    cooling_rate = 0.995
    min_temperature = 1e-10
    iterations_per_temp = 200
    
    # Run optimized simulated annealing
    while temperature > min_temperature:
        for _ in range(iterations_per_temp):
            # Make a random perturbation to a random point
            i = random.randint(0, len(current_points) - 1)
            old_point = current_points[i].copy()
            
            # Use different perturbation strategies
            if random.random() < 0.7:
                # Standard small perturbation
                perturbation = np.random.normal(0, 0.015, 2)
            else:
                # Larger perturbation for exploration
                perturbation = np.random.normal(0, 0.04, 2)
            
            new_point = old_point + perturbation
            
            # Project to triangle
            new_point = project_to_triangle(new_point)
            
            # Update point temporarily
            current_points[i] = new_point
            
            # Calculate new minimum area
            new_min_area = min_triangle_area(current_points)
            
            # Accept or reject based on simulated annealing criteria
            if new_min_area > current_min_area:
                # Always accept better solution
                current_min_area = new_min_area
                if new_min_area > best_min_area:
                    best_min_area = new_min_area
                    best_points = current_points.copy()
            else:
                # Sometimes accept worse solution based on temperature
                delta = new_min_area - current_min_area
                acceptance_probability = math.exp(delta / temperature)
                if random.random() < acceptance_probability:
                    current_min_area = new_min_area
                else:
                    # Revert if rejected
                    current_points[i] = old_point
        
        # Cool down temperature
        temperature *= cooling_rate
    
    # Add local refinement with better hill climbing
    for _ in range(300):
        # Try moving each point slightly to improve the minimum area
        for i in range(len(current_points)):
            old_point = current_points[i].copy()
            # Small perturbation
            perturbation = np.random.normal(0, 0.003, 2)
            new_point = old_point + perturbation
            new_point = project_to_triangle(new_point)
            
            current_points[i] = new_point
            new_min_area = min_triangle_area(current_points)
            
            if new_min_area <= current_min_area:
                # Revert if not better
                current_points[i] = old_point
            else:
                # Accept improvement
                current_min_area = new_min_area
                if new_min_area > best_min_area:
                    best_min_area = new_min_area
                    best_points = current_points.copy()
    
    # Final validation and return
    for i in range(len(best_points)):
        if not is_valid_point(best_points[i]):
            best_points[i] = project_to_triangle(best_points[i])
    
    return best_points
# EVOLVE-BLOCK-END
