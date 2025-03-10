#!/usr/bin/env python3
"""
Benchmark script to compare the performance of vectorized_sample_from_rles vs vectorized_sample_from_rles_fast
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from dvid_point_cloud.sampling import vectorized_sample_from_rles, vectorized_sample_from_rles_fast

def generate_synthetic_rles(num_runs, max_length=100):
    """Generate synthetic RLE data for testing"""
    # Generate random start coordinates
    starts_zyx = np.random.randint(0, 1000, size=(num_runs, 3))
    # Generate random lengths
    lengths = np.random.randint(1, max_length, size=num_runs)
    return starts_zyx, lengths

def benchmark_sampling_functions(num_runs_list, num_points=10000, repeats=5):
    """
    Benchmark both sampling functions with varying numbers of runs
    
    Args:
        num_runs_list: List of numbers of runs to test
        num_points: Number of points to sample in each test
        repeats: Number of times to repeat each test for averaging
    
    Returns:
        Dictionary with benchmark results
    """
    results = {
        'num_runs': num_runs_list,
        'fast_times': [],
        'correct_times': [],
        'speedup_factors': []
    }
    
    for num_runs in num_runs_list:
        print(f"Benchmarking with {num_runs} runs...")
        
        fast_times = []
        correct_times = []
        
        for _ in range(repeats):
            # Generate synthetic data
            starts_zyx, lengths = generate_synthetic_rles(num_runs)
            
            # Benchmark fast version
            start_time = time.time()
            _ = vectorized_sample_from_rles_fast(starts_zyx, lengths, num_points)
            fast_time = time.time() - start_time
            fast_times.append(fast_time)
            
            # Benchmark correct version
            start_time = time.time()
            _ = vectorized_sample_from_rles(starts_zyx, lengths, num_points)
            correct_time = time.time() - start_time
            correct_times.append(correct_time)
        
        # Average the times
        avg_fast = sum(fast_times) / repeats
        avg_correct = sum(correct_times) / repeats
        speedup = avg_correct / avg_fast
        
        results['fast_times'].append(avg_fast)
        results['correct_times'].append(avg_correct)
        results['speedup_factors'].append(speedup)
        
        print(f"  Fast: {avg_fast:.4f}s, Correct: {avg_correct:.4f}s, Speedup: {speedup:.2f}x")
    
    return results

def plot_results(results):
    """Plot the benchmark results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot execution times
    ax1.plot(results['num_runs'], results['fast_times'], 'o-', label='Fast (with duplicates)')
    ax1.plot(results['num_runs'], results['correct_times'], 'o-', label='Correct (unique points)')
    ax1.set_xlabel('Number of RLE runs')
    ax1.set_ylabel('Execution time (seconds)')
    ax1.set_title('Execution Time Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Plot speedup factor
    ax2.plot(results['num_runs'], results['speedup_factors'], 'o-', color='green')
    ax2.set_xlabel('Number of RLE runs')
    ax2.set_ylabel('Speedup factor (correct time / fast time)')
    ax2.set_title('Speedup Factor')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('benchmarks/sampling_benchmark_results.png')
    # Don't call plt.show() in non-interactive environments

def main():
    # Define the range of RLE runs to test
    num_runs_list = [100, 1000, 10000, 100000, 500000, 1000000]
    
    # Run benchmarks for different sample sizes
    small_results = benchmark_sampling_functions(num_runs_list, num_points=1000)
    medium_results = benchmark_sampling_functions(num_runs_list, num_points=10000)
    large_results = benchmark_sampling_functions(num_runs_list, num_points=100000)
    
    print("\nSmall Sample Size (1,000 points):")
    for i, num_runs in enumerate(small_results['num_runs']):
        print(f"  {num_runs} runs: Speedup = {small_results['speedup_factors'][i]:.2f}x")
    
    print("\nMedium Sample Size (10,000 points):")
    for i, num_runs in enumerate(medium_results['num_runs']):
        print(f"  {num_runs} runs: Speedup = {medium_results['speedup_factors'][i]:.2f}x")
    
    print("\nLarge Sample Size (100,000 points):")
    for i, num_runs in enumerate(large_results['num_runs']):
        print(f"  {num_runs} runs: Speedup = {large_results['speedup_factors'][i]:.2f}x")
    
    # Plot results for medium sample size
    plot_results(medium_results)

if __name__ == "__main__":
    main()