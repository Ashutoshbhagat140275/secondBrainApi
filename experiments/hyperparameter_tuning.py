"""
Hyperparameter Tuning Experiment for Alpha Engine

This script systematically evaluates different combinations of hyperparameters
(K, tau, beta) for the sigmoid-based alpha engine to find optimal values.

Experiments:
- K values: 25, 50, 100 (feedback scaling constant)
- tau values: 0.5, 0.6, 0.7 (confidence threshold)
- beta values: 5, 10, 20 (sigmoid sharpness)

Evaluation criteria:
1. Smooth transitions (no discontinuities)
2. Intuitive behavior (matches design goals)
3. Responsiveness to feedback and confidence changes
4. Balance between global and user heads

Usage:
    python -m experiments.hyperparameter_tuning
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.alpha_engine import compute_alpha_sigmoid


# Hyperparameter ranges to test
K_VALUES = [25, 50, 100]
TAU_VALUES = [0.5, 0.6, 0.7]
BETA_VALUES = [5, 10, 20]

# Test scenarios: (feedback_count, global_confidence, description)
TEST_SCENARIOS = [
    # New users (N=0)
    (0, 0.3, "New user, low confidence"),
    (0, 0.5, "New user, medium-low confidence"),
    (0, 0.6, "New user, medium confidence"),
    (0, 0.7, "New user, medium-high confidence"),
    (0, 0.9, "New user, high confidence"),
    
    # Early feedback (N=10-25)
    (10, 0.5, "Early feedback, medium-low confidence"),
    (10, 0.7, "Early feedback, medium-high confidence"),
    (25, 0.5, "Early feedback (25), medium-low confidence"),
    (25, 0.7, "Early feedback (25), medium-high confidence"),
    
    # Medium feedback (N=50)
    (50, 0.3, "Medium feedback, low confidence"),
    (50, 0.5, "Medium feedback, medium-low confidence"),
    (50, 0.6, "Medium feedback, medium confidence"),
    (50, 0.7, "Medium feedback, medium-high confidence"),
    (50, 0.9, "Medium feedback, high confidence"),
    
    # High feedback (N=100)
    (100, 0.3, "High feedback, low confidence"),
    (100, 0.5, "High feedback, medium-low confidence"),
    (100, 0.6, "High feedback, medium confidence"),
    (100, 0.7, "High feedback, medium-high confidence"),
    (100, 0.9, "High feedback, high confidence"),
    
    # Very high feedback (N=200)
    (200, 0.5, "Very high feedback, medium-low confidence"),
    (200, 0.7, "Very high feedback, medium-high confidence"),
    (200, 0.9, "Very high feedback, high confidence"),
]


def evaluate_hyperparameters(K: float, tau: float, beta: float) -> Dict:
    """
    Evaluate a specific hyperparameter combination across all test scenarios.
    
    Returns metrics and alpha values for analysis.
    """
    results = []
    
    for feedback_count, global_confidence, description in TEST_SCENARIOS:
        alpha_data, alpha_conf, alpha_final = compute_alpha_sigmoid(
            global_confidence=global_confidence,
            feedback_count=feedback_count,
            K=K,
            tau=tau,
            beta=beta
        )
        
        results.append({
            "feedback_count": feedback_count,
            "global_confidence": global_confidence,
            "description": description,
            "alpha_data": alpha_data,
            "alpha_conf": alpha_conf,
            "alpha_final": alpha_final,
        })
    
    # Compute metrics using numpy
    alpha_finals = [r["alpha_final"] for r in results]
    
    metrics = {
        "K": K,
        "tau": tau,
        "beta": beta,
        "mean_alpha": np.mean(alpha_finals),
        "std_alpha": np.std(alpha_finals),
        "min_alpha": np.min(alpha_finals),
        "max_alpha": np.max(alpha_finals),
        "range_alpha": np.max(alpha_finals) - np.min(alpha_finals),
        "results": results,
    }
    
    # Evaluate design goals
    metrics.update(evaluate_design_goals(results, K, tau, beta))
    
    return metrics


def evaluate_design_goals(results: List[Dict], K: float, tau: float, beta: float) -> Dict:
    """
    Evaluate how well hyperparameters meet design goals.
    
    Design goals:
    1. New users with high confidence should trust global head
    2. Users with lots of feedback should trust user head
    3. Low confidence should favor user head (if available)
    4. Smooth transitions
    """
    goals = {}
    
    # Goal 1: New user + high confidence → high alpha
    new_user_high_conf_alphas = [
        r["alpha_final"] for r in results
        if r["feedback_count"] == 0 and r["global_confidence"] >= 0.7
    ]
    new_user_high_conf = np.mean(new_user_high_conf_alphas) if new_user_high_conf_alphas else 0.0
    goals["new_user_high_conf_alpha"] = new_user_high_conf
    goals["goal1_met"] = new_user_high_conf > 0.7  # Should trust global
    
    # Goal 2: High feedback → low alpha (favor user head)
    high_feedback_alphas = [
        r["alpha_final"] for r in results
        if r["feedback_count"] >= 100
    ]
    high_feedback = np.mean(high_feedback_alphas) if high_feedback_alphas else 0.0
    goals["high_feedback_alpha"] = high_feedback
    goals["goal2_met"] = high_feedback < 0.4  # Should favor user
    
    # Goal 3: Low confidence → low alpha (favor user head if available)
    low_conf_medium_feedback_alphas = [
        r["alpha_final"] for r in results
        if r["feedback_count"] == 50 and r["global_confidence"] <= 0.5
    ]
    low_conf_medium_feedback = np.mean(low_conf_medium_feedback_alphas) if low_conf_medium_feedback_alphas else 0.0
    goals["low_conf_medium_feedback_alpha"] = low_conf_medium_feedback
    goals["goal3_met"] = low_conf_medium_feedback < 0.3  # Should favor user
    
    # Goal 4: Responsiveness to confidence changes
    # Compare alpha at N=50 with C_g=0.3 vs C_g=0.9
    alpha_low_conf = [
        r["alpha_final"] for r in results
        if r["feedback_count"] == 50 and r["global_confidence"] == 0.3
    ]
    alpha_high_conf = [
        r["alpha_final"] for r in results
        if r["feedback_count"] == 50 and r["global_confidence"] == 0.9
    ]
    
    if alpha_low_conf and alpha_high_conf:
        conf_responsiveness = alpha_high_conf[0] - alpha_low_conf[0]
        goals["confidence_responsiveness"] = conf_responsiveness
        goals["goal4_met"] = conf_responsiveness > 0.2  # Should be responsive
    else:
        goals["confidence_responsiveness"] = 0.0
        goals["goal4_met"] = False
    
    # Overall score (number of goals met)
    goals["goals_met_count"] = sum([
        goals["goal1_met"],
        goals["goal2_met"],
        goals["goal3_met"],
        goals["goal4_met"],
    ])
    
    return goals


def analyze_smoothness(K: float, tau: float, beta: float) -> Dict:
    """
    Analyze smoothness of alpha transitions.
    
    Tests small perturbations in feedback_count and global_confidence
    to ensure no discontinuities.
    """
    # Test feedback smoothness (at C_g=0.7)
    feedback_range = np.linspace(0, 200, 100)
    alpha_values_feedback = []
    
    for N in feedback_range:
        _, _, alpha = compute_alpha_sigmoid(
            global_confidence=0.7,
            feedback_count=int(N),
            K=K,
            tau=tau,
            beta=beta
        )
        alpha_values_feedback.append(alpha)
    
    # Compute max difference between consecutive points
    feedback_diffs = np.diff(alpha_values_feedback)
    max_feedback_jump = np.max(np.abs(feedback_diffs))
    
    # Test confidence smoothness (at N=50)
    confidence_range = np.linspace(0.0, 1.0, 100)
    alpha_values_confidence = []
    
    for C_g in confidence_range:
        _, _, alpha = compute_alpha_sigmoid(
            global_confidence=C_g,
            feedback_count=50,
            K=K,
            tau=tau,
            beta=beta
        )
        alpha_values_confidence.append(alpha)
    
    # Compute max difference between consecutive points
    confidence_diffs = np.diff(alpha_values_confidence)
    max_confidence_jump = np.max(np.abs(confidence_diffs))
    
    return {
        "max_feedback_jump": max_feedback_jump,
        "max_confidence_jump": max_confidence_jump,
        "is_smooth": max_feedback_jump < 0.05 and max_confidence_jump < 0.05,
    }


def run_full_experiment() -> List[Dict]:
    """
    Run full hyperparameter tuning experiment.
    
    Tests all combinations of K, tau, and beta values.
    """
    print("=" * 80)
    print("Alpha Engine Hyperparameter Tuning Experiment")
    print("=" * 80)
    print()
    print(f"Testing {len(K_VALUES)} K values: {K_VALUES}")
    print(f"Testing {len(TAU_VALUES)} tau values: {TAU_VALUES}")
    print(f"Testing {len(BETA_VALUES)} beta values: {BETA_VALUES}")
    print(f"Total combinations: {len(K_VALUES) * len(TAU_VALUES) * len(BETA_VALUES)}")
    print(f"Test scenarios: {len(TEST_SCENARIOS)}")
    print()
    
    all_results = []
    
    for K in K_VALUES:
        for tau in TAU_VALUES:
            for beta in BETA_VALUES:
                print(f"Evaluating K={K}, tau={tau}, beta={beta}...", end=" ")
                
                # Evaluate hyperparameters
                metrics = evaluate_hyperparameters(K, tau, beta)
                
                # Analyze smoothness
                smoothness = analyze_smoothness(K, tau, beta)
                metrics.update(smoothness)
                
                # Remove detailed results for summary
                del metrics["results"]
                
                all_results.append(metrics)
                
                print(f"Goals met: {metrics['goals_met_count']}/4, " +
                      f"Smooth: {metrics['is_smooth']}")
    
    print()
    print("Experiment complete!")
    print()
    
    return all_results


def print_summary(results: List[Dict]):
    """Print summary of experiment results."""
    print("=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)
    print()
    
    # Sort by goals met and smoothness
    results_sorted = sorted(
        results,
        key=lambda x: (x["goals_met_count"], x["is_smooth"]),
        reverse=True
    )
    
    print("Top 5 Hyperparameter Combinations:")
    print("-" * 80)
    
    for idx, row in enumerate(results_sorted[:5]):
        print(f"\nRank {idx + 1}:")
        print(f"  K={row['K']}, tau={row['tau']}, beta={row['beta']}")
        print(f"  Goals met: {row['goals_met_count']}/4")
        print(f"  Smooth transitions: {row['is_smooth']}")
        print(f"  Mean alpha: {row['mean_alpha']:.3f}")
        print(f"  Alpha range: [{row['min_alpha']:.3f}, {row['max_alpha']:.3f}]")
        print(f"  Confidence responsiveness: {row['confidence_responsiveness']:.3f}")
    
    print()
    print("=" * 80)
    print("GOAL ANALYSIS")
    print("=" * 80)
    print()
    
    # Analyze each goal
    print("Goal 1: New user + high confidence → high alpha (>0.7)")
    goal1_met = [r for r in results if r["goal1_met"]]
    print(f"  Met by {len(goal1_met)}/{len(results)} combinations")
    if goal1_met:
        best = goal1_met[0]
        print(f"  Best: K={best['K']}, tau={best['tau']}, " +
              f"beta={best['beta']} " +
              f"(alpha={best['new_user_high_conf_alpha']:.3f})")
    
    print()
    print("Goal 2: High feedback → low alpha (<0.4)")
    goal2_met = [r for r in results if r["goal2_met"]]
    print(f"  Met by {len(goal2_met)}/{len(results)} combinations")
    if goal2_met:
        best = goal2_met[0]
        print(f"  Best: K={best['K']}, tau={best['tau']}, " +
              f"beta={best['beta']} " +
              f"(alpha={best['high_feedback_alpha']:.3f})")
    
    print()
    print("Goal 3: Low confidence + medium feedback → low alpha (<0.3)")
    goal3_met = [r for r in results if r["goal3_met"]]
    print(f"  Met by {len(goal3_met)}/{len(results)} combinations")
    if goal3_met:
        best = goal3_met[0]
        print(f"  Best: K={best['K']}, tau={best['tau']}, " +
              f"beta={best['beta']} " +
              f"(alpha={best['low_conf_medium_feedback_alpha']:.3f})")
    
    print()
    print("Goal 4: Responsive to confidence changes (>0.2 difference)")
    goal4_met = [r for r in results if r["goal4_met"]]
    print(f"  Met by {len(goal4_met)}/{len(results)} combinations")
    if goal4_met:
        best_responsive = max(goal4_met, key=lambda x: x["confidence_responsiveness"])
        print(f"  Best: K={best_responsive['K']}, tau={best_responsive['tau']}, " +
              f"beta={best_responsive['beta']} " +
              f"(responsiveness={best_responsive['confidence_responsiveness']:.3f})")
    
    print()
    print("=" * 80)
    print("SMOOTHNESS ANALYSIS")
    print("=" * 80)
    print()
    
    smooth_combinations = [r for r in results if r["is_smooth"]]
    print(f"Smooth transitions: {len(smooth_combinations)}/{len(results)} combinations")
    
    if smooth_combinations:
        print("\nSmooth combinations:")
        for row in smooth_combinations:
            print(f"  K={row['K']}, tau={row['tau']}, beta={row['beta']} " +
                  f"(max jumps: feedback={row['max_feedback_jump']:.4f}, " +
                  f"confidence={row['max_confidence_jump']:.4f})")
    
    print()


def print_recommendations(results: List[Dict]):
    """Print recommendations based on experiment results."""
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    # Find best overall combination
    results_sorted = sorted(
        results,
        key=lambda x: (x["goals_met_count"], x["is_smooth"], x["confidence_responsiveness"]),
        reverse=True
    )
    
    best = results_sorted[0]
    
    print("RECOMMENDED HYPERPARAMETERS:")
    print(f"  K = {best['K']}")
    print(f"  tau = {best['tau']}")
    print(f"  beta = {best['beta']}")
    print()
    print("Rationale:")
    print(f"  - Meets {best['goals_met_count']}/4 design goals")
    print(f"  - Smooth transitions: {best['is_smooth']}")
    print(f"  - Confidence responsiveness: {best['confidence_responsiveness']:.3f}")
    print(f"  - Mean alpha: {best['mean_alpha']:.3f}")
    print(f"  - Alpha range: [{best['min_alpha']:.3f}, {best['max_alpha']:.3f}]")
    print()
    
    # Compare with current defaults
    current = [r for r in results if r["K"] == 50 and r["tau"] == 0.6 and r["beta"] == 10]
    
    if current:
        current = current[0]
        print("COMPARISON WITH CURRENT DEFAULTS (K=50, tau=0.6, beta=10):")
        print(f"  Goals met: {current['goals_met_count']}/4 (recommended: {best['goals_met_count']}/4)")
        print(f"  Smooth: {current['is_smooth']} (recommended: {best['is_smooth']})")
        print(f"  Responsiveness: {current['confidence_responsiveness']:.3f} " +
              f"(recommended: {best['confidence_responsiveness']:.3f})")
        print()
        
        if (best['K'] == 50 and best['tau'] == 0.6 and best['beta'] == 10):
            print("✓ Current defaults are optimal!")
        else:
            print("→ Consider updating to recommended values for improved performance")
    
    print()
    
    # K value analysis
    print("K VALUE ANALYSIS:")
    for K in K_VALUES:
        k_results = [r for r in results if r["K"] == K]
        avg_goals = np.mean([r["goals_met_count"] for r in k_results])
        print(f"  K={K}: Average {avg_goals:.1f}/4 goals met")
    print()
    
    # tau value analysis
    print("TAU VALUE ANALYSIS:")
    for tau in TAU_VALUES:
        tau_results = [r for r in results if r["tau"] == tau]
        avg_goals = np.mean([r["goals_met_count"] for r in tau_results])
        print(f"  tau={tau}: Average {avg_goals:.1f}/4 goals met")
    print()
    
    # beta value analysis
    print("BETA VALUE ANALYSIS:")
    for beta in BETA_VALUES:
        beta_results = [r for r in results if r["beta"] == beta]
        avg_goals = np.mean([r["goals_met_count"] for r in beta_results])
        avg_responsiveness = np.mean([r["confidence_responsiveness"] for r in beta_results])
        print(f"  beta={beta}: Average {avg_goals:.1f}/4 goals met, " +
              f"responsiveness={avg_responsiveness:.3f}")
    print()


def save_results(results: List[Dict], filename: str = "hyperparameter_tuning_results.json"):
    """Save results to JSON file."""
    output_path = Path(__file__).parent / filename
    
    # Convert numpy types to native Python types for JSON serialization
    results_serializable = []
    for r in results:
        r_copy = {}
        for key, value in r.items():
            if isinstance(value, (np.floating, np.integer)):
                r_copy[key] = float(value)
            elif isinstance(value, (bool, np.bool_)):
                r_copy[key] = bool(value)
            else:
                r_copy[key] = value
        results_serializable.append(r_copy)
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    print()


if __name__ == "__main__":
    # Run experiment
    results = run_full_experiment()
    
    # Print analysis
    print_summary(results)
    print_recommendations(results)
    
    # Save results
    save_results(results)
    
    print("=" * 80)
    print("Experiment complete! Review results above.")
    print("=" * 80)
