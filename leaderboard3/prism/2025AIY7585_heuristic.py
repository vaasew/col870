GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START
GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    
    Approach: Multi-start local search with different model orderings.
    Instead of one greedy pass, we try multiple greedy strategies with different 
    model orderings, then use local search to improve the solutions by swapping
    models between GPUs. This fundamentally differs from the single greedy pass
    by exploring a wider solution space and refining the result.
    
    Args:
        gpu_num: Number of GPUs
        models: List of models to place, each with model_size, req_rate, and slo attributes
        
    Returns:
        A placement dictionary mapping GPU IDs to lists of models
    """
    
    def greedy_placement(model_ordering):
        """Perform greedy assignment with a specific model ordering"""
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        gpu_memory = [GPU_MEM_SIZE for _ in range(gpu_num)]
        gpu_req_rate_sum = [0.0 for _ in range(gpu_num)]
        
        for model in model_ordering:
            best_gpu = None
            min_new_kvpr = float('inf')
            
            for gpu_id in range(gpu_num):
                if model.model_size <= gpu_memory[gpu_id]:
                    numerator = gpu_req_rate_sum[gpu_id] + (model.req_rate / model.slo)
                    denominator = gpu_memory[gpu_id] - model.model_size
                    
                    if denominator > 0:
                        new_kvpr = numerator / denominator
                        if new_kvpr < min_new_kvpr:
                            min_new_kvpr = new_kvpr
                            best_gpu = gpu_id
            
            if best_gpu is None:
                return None  # Cannot place this model
            
            placement[best_gpu].append(model)
            gpu_memory[best_gpu] -= model.model_size
            gpu_req_rate_sum[best_gpu] += model.req_rate / model.slo
        
        return placement
    
    def evaluate_placement(placement):
        """Calculate maximum KVPR for a placement"""
        max_kvpr = 0.0
        for gpu_id in range(gpu_num):
            models = placement[gpu_id]
            if not models:
                continue
            req_rate_sum = sum(m.req_rate / m.slo for m in models)
            mem_used = sum(m.model_size for m in models)
            if mem_used < GPU_MEM_SIZE:
                kvpr = req_rate_sum / (GPU_MEM_SIZE - mem_used)
                max_kvpr = max(max_kvpr, kvpr)
        return max_kvpr
    
    def local_search(placement):
        """Try to improve placement by swapping models between GPUs"""
        improved = True
        while improved:
            improved = False
            max_kvpr_before = evaluate_placement(placement)
            
            # Try swapping models between GPUs
            for gpu1 in range(gpu_num):
                for gpu2 in range(gpu_num):
                    if gpu1 == gpu2:
                        continue
                    models1 = placement[gpu1]
                    models2 = placement[gpu2]
                    
                    if not models1 or not models2:
                        continue
                    
                    # Try swapping each pair of models
                    for i, m1 in enumerate(models1):
                        for j, m2 in enumerate(models2):
                            # Check if swap is valid (both models fit)
                            if (m1.model_size <= GPU_MEM_SIZE - sum(m.model_size for m in models2[:j] + models2[j+1:] + [m2]) and
                                m2.model_size <= GPU_MEM_SIZE - sum(m.model_size for m in models1[:i] + models1[i+1:] + [m1])):
                                
                                # Perform swap
                                new_placement = {gpu_id: [] for gpu_id in range(gpu_num)}
                                for k in range(gpu_num):
                                    if k == gpu1:
                                        new_placement[k] = models1[:i] + [m2] + models1[i+1:]
                                    elif k == gpu2:
                                        new_placement[k] = models2[:j] + [m1] + models2[j+1:]
                                    else:
                                        new_placement[k] = placement[k]
                                
                                new_kvpr = evaluate_placement(new_placement)
                                if new_kvpr < max_kvpr_before:
                                    placement.update(new_placement)
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return placement
    
    # Generate multiple different orderings
    orderings = []
    
    # Original ordering (by req_rate/slo)
    orderings.append(sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True))
    
    # Ordering by model_size (largest first)
    orderings.append(sorted(models, key=lambda m: m.model_size, reverse=True))
    
    # Ordering by req_rate (highest first)
    orderings.append(sorted(models, key=lambda m: m.req_rate, reverse=True))
    
    # Ordering by model_size + req_rate (combined)
    orderings.append(sorted(models, key=lambda m: m.model_size * m.req_rate, reverse=True))
    
    # Random ordering
    import random
    shuffled_models = models.copy()
    random.shuffle(shuffled_models)
    orderings.append(shuffled_models)
    
    best_placement = None
    best_kvpr = float('inf')
    
    # Try all orderings
    for model_ordering in orderings:
        placement = greedy_placement(model_ordering)
        if placement is not None:
            # Apply local search to improve the result
            improved_placement = local_search(placement.copy())
            kvpr = evaluate_placement(improved_placement)
            
            if kvpr < best_kvpr:
                best_kvpr = kvpr
                best_placement = improved_placement
    
    if best_placement is None:
        raise ValueError("Unable to find valid placement for any ordering")
    
    return best_placement
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    # Test the algorithm

    import numpy as np
    from evaluator import calculate_kvcache_pressure, generate_test_gpu_models, safe_float

    test_cases = generate_test_gpu_models()
    all_kvpr = []
    for i, (gpu_num, gpu_models) in enumerate(test_cases):

        results = compute_model_placement(gpu_num, gpu_models)
        max_kvpr = calculate_kvcache_pressure(results)
        all_kvpr.append(safe_float(max_kvpr))

    avg_kvpr = np.mean(all_kvpr)
    if avg_kvpr != 0:
        avg_kvpr = 1.0 / avg_kvpr

    print(f"Max KVPR: {avg_kvpr:.3f}")
