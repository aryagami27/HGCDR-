
import numpy as np
from scipy import stats

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculates the bootstrap confidence interval.
    Args:
        data: List or array of metric scores (e.g., HR@10 from multiple runs or users).
        confidence: Confidence level (default 0.95).
    """
    data = np.array(data)
    n = len(data)
    m = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def paired_t_test(baseline_scores, model_scores):
    """
    Performs a paired t-test between baseline and new model.
    Args:
        baseline_scores: List/Array of scores for baseline.
        model_scores: List/Array of scores for new model.
    Returns:
        t_statistic, p_value
    """
    t_stat, p_val = stats.ttest_rel(baseline_scores, model_scores)
    return t_stat, p_val

def print_significance(baseline_name, baseline_res, model_name, model_res):
    """
    Prints significance report.
    """
    print(f"\n--- Statistical Significance ({baseline_name} vs {model_name}) ---")
    mean_b, low_b, high_b = calculate_confidence_interval(baseline_res)
    mean_m, low_m, high_m = calculate_confidence_interval(model_res)
    
    print(f"{baseline_name}: {mean_b:.4f} ± {mean_b - low_b:.4f} (95% CI: [{low_b:.4f}, {high_b:.4f}])")
    print(f"{model_name}: {mean_m:.4f} ± {mean_m - low_m:.4f} (95% CI: [{low_m:.4f}, {high_m:.4f}])")
    
    if len(baseline_res) == len(model_res):
        t, p = paired_t_test(baseline_res, model_res)
        print(f"Paired t-test: t={t:.4f}, p={p:.4e}")
        if p < 0.05:
            print("Result: Statistically Significant (p < 0.05)")
        else:
            print("Result: Not Significant")
    else:
        print("Warning: Sample sizes differ, skipping paired t-test.")
