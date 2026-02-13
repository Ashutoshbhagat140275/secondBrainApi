"""
Performance benchmark tests for the alpha engine module.

Tests verify that the sigmoid-based alpha computation meets performance requirements:
- Measure computation time for sigmoid vs linear formulas
- Verify sigmoid adds < 1ms overhead compared to linear
- Test with 1000 iterations to get reliable measurements

Validates Requirement: 7.3 (Performance: Alpha computation adds < 1ms overhead)
"""

import pytest
import time
import numpy as np
from typing import List, Dict

from app.services.alpha_engine import (
    compute_alpha_sigmoid,
    compute_alpha_linear,
    compute_blend_weight
)


class TestAlphaEnginePerformance:
    """Performance benchmark tests for alpha engine (Requirement 7.3)."""
    
    def test_sigmoid_computation_time(self):
        """Measure sigmoid alpha computation time over 1000 iterations."""
        # Test parameters
        iterations = 1000
        feedback_count = 50
        global_confidence = 0.7
        
        # Warm-up (to avoid cold start effects)
        for _ in range(10):
            compute_alpha_sigmoid(global_confidence, feedback_count)
        
        # Measure time for 1000 iterations
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            compute_alpha_sigmoid(global_confidence, feedback_count)
        
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time_per_call = total_time / iterations
        avg_time_ms = avg_time_per_call * 1000
        
        # Log results
        print(f"\nSigmoid formula performance:")
        print(f"  Total time: {total_time:.6f}s")
        print(f"  Average per call: {avg_time_per_call:.6f}s ({avg_time_ms:.4f}ms)")
        print(f"  Iterations: {iterations}")
        
        # Verify reasonable performance (should be much faster than 1ms)
        assert avg_time_ms < 1.0, \
            f"Sigmoid computation too slow: {avg_time_ms:.4f}ms per call"
    
    def test_linear_computation_time(self):
        """Measure linear alpha computation time over 1000 iterations."""
        # Test parameters
        iterations = 1000
        feedback_count = 50
        global_confidence = 0.7
        
        # Warm-up
        for _ in range(10):
            compute_alpha_linear(global_confidence, feedback_count)
        
        # Measure time for 1000 iterations
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            compute_alpha_linear(global_confidence, feedback_count)
        
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time_per_call = total_time / iterations
        avg_time_ms = avg_time_per_call * 1000
        
        # Log results
        print(f"\nLinear formula performance:")
        print(f"  Total time: {total_time:.6f}s")
        print(f"  Average per call: {avg_time_per_call:.6f}s ({avg_time_ms:.4f}ms)")
        print(f"  Iterations: {iterations}")
        
        # Verify reasonable performance
        assert avg_time_ms < 1.0, \
            f"Linear computation too slow: {avg_time_ms:.4f}ms per call"
    
    def test_sigmoid_vs_linear_overhead(self):
        """
        Compare sigmoid vs linear computation time and verify overhead < 1ms.
        
        This is the key performance requirement: sigmoid should add < 1ms overhead
        compared to the linear formula.
        
        Validates Requirement 7.3.
        """
        iterations = 1000
        feedback_count = 50
        global_confidence = 0.7
        
        # Warm-up both functions
        for _ in range(10):
            compute_alpha_sigmoid(global_confidence, feedback_count)
            compute_alpha_linear(global_confidence, feedback_count)
        
        # Measure linear formula time
        start_linear = time.perf_counter()
        for _ in range(iterations):
            compute_alpha_linear(global_confidence, feedback_count)
        end_linear = time.perf_counter()
        linear_time = end_linear - start_linear
        
        # Measure sigmoid formula time
        start_sigmoid = time.perf_counter()
        for _ in range(iterations):
            compute_alpha_sigmoid(global_confidence, feedback_count)
        end_sigmoid = time.perf_counter()
        sigmoid_time = end_sigmoid - start_sigmoid
        
        # Calculate overhead
        overhead_total = sigmoid_time - linear_time
        overhead_per_call = overhead_total / iterations
        overhead_ms = overhead_per_call * 1000
        
        # Calculate average times
        linear_avg_ms = (linear_time / iterations) * 1000
        sigmoid_avg_ms = (sigmoid_time / iterations) * 1000
        
        # Log comparison
        print(f"\nPerformance comparison (1000 iterations):")
        print(f"  Linear:  {linear_time:.6f}s ({linear_avg_ms:.4f}ms per call)")
        print(f"  Sigmoid: {sigmoid_time:.6f}s ({sigmoid_avg_ms:.4f}ms per call)")
        print(f"  Overhead: {overhead_total:.6f}s ({overhead_ms:.4f}ms per call)")
        
        if overhead_ms > 0:
            slowdown_pct = (overhead_ms / linear_avg_ms) * 100
            print(f"  Slowdown: {slowdown_pct:.1f}%")
        else:
            print(f"  Slowdown: 0% (sigmoid is faster or equal)")
        
        # Verify requirement: sigmoid adds < 1ms overhead
        assert overhead_ms < 1.0, \
            f"Sigmoid overhead too high: {overhead_ms:.4f}ms per call (requirement: < 1ms)"
    
    def test_performance_with_varying_inputs(self):
        """Test performance across different input ranges."""
        iterations = 1000
        
        # Test different scenarios
        scenarios = [
            ("New user, low confidence", 0, 0.3),
            ("New user, high confidence", 0, 0.9),
            ("Medium feedback, medium confidence", 50, 0.6),
            ("High feedback, low confidence", 200, 0.3),
            ("High feedback, high confidence", 200, 0.9),
        ]
        
        results = []
        
        for description, feedback_count, global_confidence in scenarios:
            # Warm-up
            for _ in range(10):
                compute_alpha_sigmoid(global_confidence, feedback_count)
            
            # Measure
            start_time = time.perf_counter()
            for _ in range(iterations):
                compute_alpha_sigmoid(global_confidence, feedback_count)
            end_time = time.perf_counter()
            
            avg_time_ms = ((end_time - start_time) / iterations) * 1000
            results.append((description, avg_time_ms))
            
            # Verify performance for each scenario
            assert avg_time_ms < 1.0, \
                f"{description}: computation too slow ({avg_time_ms:.4f}ms)"
        
        # Log all results
        print(f"\nPerformance across scenarios (1000 iterations each):")
        for description, avg_time_ms in results:
            print(f"  {description}: {avg_time_ms:.4f}ms per call")
    
    def test_compute_blend_weight_performance(self):
        """Test performance of the main entry point compute_blend_weight."""
        iterations = 1000
        feedback_count = 50
        global_confidence = 0.7
        
        # Warm-up
        for _ in range(10):
            compute_blend_weight(global_confidence, feedback_count)
        
        # Measure time
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            result = compute_blend_weight(global_confidence, feedback_count)
        
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time_per_call = total_time / iterations
        avg_time_ms = avg_time_per_call * 1000
        
        # Log results
        print(f"\ncompute_blend_weight performance:")
        print(f"  Total time: {total_time:.6f}s")
        print(f"  Average per call: {avg_time_per_call:.6f}s ({avg_time_ms:.4f}ms)")
        print(f"  Iterations: {iterations}")
        
        # Verify performance
        assert avg_time_ms < 1.0, \
            f"compute_blend_weight too slow: {avg_time_ms:.4f}ms per call"
    
    def test_batch_performance(self):
        """Test performance when computing alpha for multiple users in batch."""
        num_users = 100
        iterations = 10  # 10 batches of 100 users = 1000 total computations
        
        # Generate random test data
        np.random.seed(42)
        feedback_counts = np.random.randint(0, 200, size=num_users)
        confidence_values = np.random.uniform(0.3, 0.9, size=num_users)
        
        # Warm-up
        for i in range(10):
            compute_alpha_sigmoid(confidence_values[i % num_users], 
                                int(feedback_counts[i % num_users]))
        
        # Measure batch processing time
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            for i in range(num_users):
                compute_alpha_sigmoid(
                    float(confidence_values[i]),
                    int(feedback_counts[i])
                )
        
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_computations = num_users * iterations
        total_time = end_time - start_time
        avg_time_per_call = total_time / total_computations
        avg_time_ms = avg_time_per_call * 1000
        
        # Log results
        print(f"\nBatch performance ({num_users} users × {iterations} iterations):")
        print(f"  Total computations: {total_computations}")
        print(f"  Total time: {total_time:.6f}s")
        print(f"  Average per call: {avg_time_per_call:.6f}s ({avg_time_ms:.4f}ms)")
        print(f"  Throughput: {total_computations / total_time:.0f} computations/second")
        
        # Verify performance
        assert avg_time_ms < 1.0, \
            f"Batch computation too slow: {avg_time_ms:.4f}ms per call"
    
    def test_worst_case_performance(self):
        """Test performance with worst-case inputs (extreme values)."""
        iterations = 1000
        
        # Worst-case scenarios that might trigger edge cases
        worst_cases = [
            ("Very high feedback", 1_000_000, 0.5),
            ("Confidence at 0", 50, 0.0),
            ("Confidence at 1", 50, 1.0),
            ("Both extreme", 1_000_000, 1.0),
        ]
        
        for description, feedback_count, global_confidence in worst_cases:
            # Warm-up
            for _ in range(10):
                compute_alpha_sigmoid(global_confidence, feedback_count)
            
            # Measure
            start_time = time.perf_counter()
            for _ in range(iterations):
                compute_alpha_sigmoid(global_confidence, feedback_count)
            end_time = time.perf_counter()
            
            avg_time_ms = ((end_time - start_time) / iterations) * 1000
            
            print(f"\nWorst-case: {description}")
            print(f"  Average time: {avg_time_ms:.4f}ms per call")
            
            # Verify performance even in worst cases
            assert avg_time_ms < 1.0, \
                f"{description}: computation too slow ({avg_time_ms:.4f}ms)"
    
    def test_performance_consistency(self):
        """Test that performance is consistent across multiple runs."""
        iterations = 1000
        num_runs = 5
        feedback_count = 50
        global_confidence = 0.7
        
        run_times = []
        
        for run in range(num_runs):
            # Warm-up
            for _ in range(10):
                compute_alpha_sigmoid(global_confidence, feedback_count)
            
            # Measure
            start_time = time.perf_counter()
            for _ in range(iterations):
                compute_alpha_sigmoid(global_confidence, feedback_count)
            end_time = time.perf_counter()
            
            avg_time_ms = ((end_time - start_time) / iterations) * 1000
            run_times.append(avg_time_ms)
        
        # Calculate statistics
        mean_time = np.mean(run_times)
        std_time = np.std(run_times)
        min_time = np.min(run_times)
        max_time = np.max(run_times)
        
        # Log results
        print(f"\nPerformance consistency ({num_runs} runs × {iterations} iterations):")
        print(f"  Mean: {mean_time:.4f}ms")
        print(f"  Std:  {std_time:.4f}ms")
        print(f"  Min:  {min_time:.4f}ms")
        print(f"  Max:  {max_time:.4f}ms")
        print(f"  Range: {max_time - min_time:.4f}ms")
        
        # Verify all runs meet performance requirement
        for i, run_time in enumerate(run_times):
            assert run_time < 1.0, \
                f"Run {i+1} too slow: {run_time:.4f}ms per call"
        
        # Verify consistency (std should be small relative to mean)
        cv = (std_time / mean_time) * 100  # Coefficient of variation
        print(f"  Coefficient of variation: {cv:.1f}%")
        
        # Performance should be reasonably consistent (CV < 50%)
        assert cv < 50, \
            f"Performance too inconsistent: CV={cv:.1f}%"


class TestPerformanceSummary:
    """Summary test that runs all benchmarks and reports overall results."""
    
    def test_performance_summary(self):
        """
        Run comprehensive performance benchmarks and generate summary report.
        
        This test provides a complete performance profile of the alpha engine.
        """
        iterations = 1000
        
        # Test scenarios
        scenarios = [
            ("Linear formula", lambda: compute_alpha_linear(0.7, 50)),
            ("Sigmoid formula", lambda: compute_alpha_sigmoid(0.7, 50)),
            ("Blend weight (current config)", lambda: compute_blend_weight(0.7, 50)),
        ]
        
        results = {}
        
        print("\n" + "="*70)
        print("ALPHA ENGINE PERFORMANCE BENCHMARK SUMMARY")
        print("="*70)
        
        for name, func in scenarios:
            # Warm-up
            for _ in range(10):
                func()
            
            # Measure
            start_time = time.perf_counter()
            for _ in range(iterations):
                func()
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            avg_time_ms = (total_time / iterations) * 1000
            throughput = iterations / total_time
            
            results[name] = {
                'total_time': total_time,
                'avg_time_ms': avg_time_ms,
                'throughput': throughput
            }
            
            print(f"\n{name}:")
            print(f"  Total time:    {total_time:.6f}s")
            print(f"  Avg per call:  {avg_time_ms:.6f}ms")
            print(f"  Throughput:    {throughput:.0f} calls/second")
        
        # Calculate overhead
        if 'Linear formula' in results and 'Sigmoid formula' in results:
            linear_time = results['Linear formula']['avg_time_ms']
            sigmoid_time = results['Sigmoid formula']['avg_time_ms']
            overhead = sigmoid_time - linear_time
            
            print(f"\n" + "-"*70)
            print(f"Sigmoid vs Linear Comparison:")
            print(f"  Overhead: {overhead:.6f}ms per call")
            
            if overhead > 0:
                slowdown_pct = (overhead / linear_time) * 100
                print(f"  Slowdown: {slowdown_pct:.1f}%")
            else:
                print(f"  Slowdown: 0% (sigmoid is faster or equal)")
            
            # Verify requirement
            assert overhead < 1.0, \
                f"REQUIREMENT FAILED: Sigmoid overhead {overhead:.4f}ms exceeds 1ms limit"
            
            print(f"  ✓ Requirement met: overhead < 1ms")
        
        print("\n" + "="*70)
        print(f"All performance tests passed with {iterations} iterations")
        print("="*70 + "\n")
