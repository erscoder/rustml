#!/usr/bin/env python3
"""
Validate rustml stats against scipy.
Generates ground truth for testing.
"""

import json
import subprocess
import sys

import numpy as np
from scipy import stats

# Generate test datasets with various characteristics
np.random.seed(42)

def generate_datasets():
    """Generate 25+ test datasets with varying properties."""
    datasets = []
    
    # 1. Basic normal distributions
    for i in range(5):
        n = 10 + i * 20
        x = np.random.randn(n) * (1 + i * 0.5) + (i * 2)
        y = np.random.randn(n) * (1 + i * 0.5) + (i * 2 + 0.5)
        datasets.append({"name": f"normal_{n}_{i}", "x": x.tolist(), "y": y.tolist()})
    
    # 2. Extreme p-value cases (small)
    for df in [2, 5, 10, 20, 50]:
        t = np.linspace(-5, 5, 100)
        p = 2 * (1 - stats.t.cdf(np.abs(t), df))
        datasets.append({
            "name": f"small_p_t_df{df}",
            "t_stat": t.tolist(),
            "df": df,
            "p_expected": p.tolist()
        })
    
    # 3. Large samples
    for n in [1000, 5000, 10000]:
        x = np.random.randn(n)
        y = np.random.randn(n) + 0.1
        datasets.append({"name": f"large_n_{n}", "x": x.tolist(), "y": y.tolist()})
    
    # 4. Edge cases: small samples
    for n in [3, 4, 5]:
        x = np.random.randn(n)
        y = np.random.randn(n)
        datasets.append({"name": f"small_n_{n}", "x": x.tolist(), "y": y.tolist()})
    
    # 5. Correlation test cases
    for r in [0.0, 0.3, 0.5, 0.7, 0.9, 0.99]:
        n = 50
        x = np.random.randn(n)
        y = r * x + np.sqrt(1 - r**2) * np.random.randn(n)
        datasets.append({"name": f"corr_r{r}", "x": x.tolist(), "y": y.tolist()})
    
    return datasets

def compute_ground_truth(datasets):
    """Compute expected values using scipy."""
    results = []
    
    for ds in datasets:
        name = ds["name"]
        
        # Skip t-test specific datasets
        if "t_stat" in ds:
            result = {"name": name}
            t = np.array(ds["t_stat"])
            df = ds["df"]
            p = 2 * (1 - stats.t.cdf(np.abs(t), df))
            result["p_expected"] = p.tolist()
            results.append(result)
            continue
        
        x = np.array(ds["x"])
        y = np.array(ds.get("y", x))
        
        result = {"name": name}
        
        # Descriptive stats
        result["mean"] = float(np.mean(x))
        result["std"] = float(np.std(x, ddof=1))
        result["var"] = float(np.var(x, ddof=1))
        result["median"] = float(np.median(x))
        
        # Hypothesis tests
        t_stat, p_val = stats.ttest_ind(x, y)
        result["ttest_ind_t"] = float(t_stat)
        result["ttest_ind_p"] = float(p_val)
        
        t_stat_1, p_val_1 = stats.ttest_1samp(x, 0.0)
        result["ttest_1samp_t"] = float(t_stat_1)
        result["ttest_1samp_p"] = float(p_val_1)
        
        r, p = stats.pearsonr(x, y)
        result["pearsonr_r"] = float(r)
        result["pearsonr_p"] = float(p)
        
        results.append(result)
    
    return results

def main():
    print("Generating test datasets...")
    datasets = generate_datasets()
    
    print("Computing ground truth with scipy...")
    results = compute_ground_truth(datasets)
    
    # Save to JSON
    output = {
        "datasets": datasets,
        "ground_truth": results,
        "tolerance": 1e-6
    }
    
    with open("benchmarks/validation_data.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Generated {len(datasets)} test cases")
    print("Saved to benchmarks/validation_data.json")
    
    # Print summary
    print("\n=== Validation Data Summary ===")
    for r in results[:5]:
        print(f"{r['name']}: pearsonr_p={r['pearsonr_p']:.2e}, ttest_ind_p={r['ttest_ind_p']:.2e}")
    print("...")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
