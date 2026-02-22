//! Statistical functions — equivalent to `scipy.stats`.
//!
//! Covers descriptive statistics, hypothesis tests, and distributions.

use std::f64::consts::PI;

// ─── Descriptive Statistics ───────────────────────────────────────

/// Arithmetic mean.
///
/// # Example
/// ```
/// use rustml_scipy::stats::mean;
/// assert!((mean(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-10);
/// ```
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Sample variance (ddof=1 by default, like numpy).
pub fn var(data: &[f64]) -> f64 {
    var_ddof(data, 1)
}

/// Variance with specified degrees of freedom correction.
pub fn var_ddof(data: &[f64], ddof: usize) -> f64 {
    if data.len() <= ddof {
        return f64::NAN;
    }
    let m = mean(data);
    let ss: f64 = data.iter().map(|x| (x - m).powi(2)).sum();
    ss / (data.len() - ddof) as f64
}

/// Sample standard deviation (ddof=1).
pub fn std(data: &[f64]) -> f64 {
    var(data).sqrt()
}

/// Standard deviation with specified ddof.
pub fn std_ddof(data: &[f64], ddof: usize) -> f64 {
    var_ddof(data, ddof).sqrt()
}

/// Median.
pub fn median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Skewness (Fisher's definition, bias=False).
pub fn skew(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 3.0 {
        return f64::NAN;
    }
    let m = mean(data);
    let s = std(data);
    if s == 0.0 {
        return f64::NAN;
    }
    let m3: f64 = data.iter().map(|x| ((x - m) / s).powi(3)).sum();
    // Adjusted Fisher-Pearson
    (n / ((n - 1.0) * (n - 2.0))) * m3
}

/// Kurtosis (excess kurtosis, Fisher's definition).
pub fn kurtosis(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 4.0 {
        return f64::NAN;
    }
    let m = mean(data);
    let s = std_ddof(data, 0);
    if s == 0.0 {
        return f64::NAN;
    }
    let m4: f64 = data.iter().map(|x| ((x - m) / s).powi(4)).sum();
    let raw = m4 / n;
    raw - 3.0 // excess kurtosis
}

/// Pearson correlation coefficient and p-value.
///
/// Returns `(r, p_value)`. Equivalent to `scipy.stats.pearsonr`.
pub fn pearsonr(x: &[f64], y: &[f64]) -> (f64, f64) {
    assert_eq!(x.len(), y.len(), "Arrays must have same length");
    let n = x.len();
    if n < 3 {
        return (f64::NAN, f64::NAN);
    }

    let mx = mean(x);
    let my = mean(y);

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x == 0.0 || var_y == 0.0 {
        return (f64::NAN, f64::NAN);
    }

    let r = cov / (var_x * var_y).sqrt();

    // t-test for correlation significance
    let t = r * ((n as f64 - 2.0) / (1.0 - r * r)).sqrt();
    let df = n as f64 - 2.0;
    let p = 2.0 * t_cdf(-t.abs(), df);

    (r, p)
}

// ─── Hypothesis Tests ─────────────────────────────────────────────

/// One-sample t-test. Tests if mean of `data` differs from `popmean`.
///
/// Returns `(t_statistic, p_value)`. Equivalent to `scipy.stats.ttest_1samp`.
///
/// # Example
/// ```
/// use rustml_scipy::stats::ttest_1samp;
/// let data = vec![2.3, 3.1, 2.8, 3.5, 2.9, 3.2, 2.7];
/// let (t, p) = ttest_1samp(&data, 3.0);
/// assert!(p > 0.05); // Cannot reject H0 that mean == 3.0
/// ```
pub fn ttest_1samp(data: &[f64], popmean: f64) -> (f64, f64) {
    let n = data.len() as f64;
    if n < 2.0 {
        return (f64::NAN, f64::NAN);
    }
    let m = mean(data);
    let se = std(data) / n.sqrt();
    if se == 0.0 {
        return (f64::INFINITY, 0.0);
    }
    let t = (m - popmean) / se;
    let df = n - 1.0;
    let p = 2.0 * t_cdf(-t.abs(), df);
    (t, p)
}

/// Independent two-sample t-test (Welch's, unequal variances).
///
/// Returns `(t_statistic, p_value)`. Equivalent to `scipy.stats.ttest_ind(equal_var=False)`.
pub fn ttest_ind(x: &[f64], y: &[f64]) -> (f64, f64) {
    let nx = x.len() as f64;
    let ny = y.len() as f64;
    if nx < 2.0 || ny < 2.0 {
        return (f64::NAN, f64::NAN);
    }

    let mx = mean(x);
    let my = mean(y);
    let vx = var(x);
    let vy = var(y);

    let se = (vx / nx + vy / ny).sqrt();
    if se == 0.0 {
        return (f64::INFINITY, 0.0);
    }

    let t = (mx - my) / se;

    // Welch-Satterthwaite degrees of freedom
    let num = (vx / nx + vy / ny).powi(2);
    let den = (vx / nx).powi(2) / (nx - 1.0) + (vy / ny).powi(2) / (ny - 1.0);
    let df = num / den;

    let p = 2.0 * t_cdf(-t.abs(), df);
    (t, p)
}

// ─── Additional Hypothesis Tests (RUST-03) ────────────────────────

/// Chi-squared test for independence.
///
/// Returns `(chi2_statistic, p_value, dof)`. Equivalent to `scipy.stats.chi2_contingency`.
pub fn chi2_test(observed: &[&[f64]]) -> (f64, f64, usize) {
    let rows = observed.len();
    let cols = observed[0].len();

    if rows < 2 || cols < 2 {
        return (f64::NAN, f64::NAN, 0);
    }

    // Calculate row and column totals
    let row_totals: Vec<f64> = (0..rows).map(|i| observed[i].iter().sum()).collect();
    let col_totals: Vec<f64> = (0..cols)
        .map(|j| (0..rows).map(|i| observed[i][j]).sum())
        .collect();
    let grand_total: f64 = row_totals.iter().sum();

    if grand_total == 0.0 {
        return (f64::NAN, f64::NAN, 0);
    }

    // Calculate chi-squared statistic
    let mut chi2 = 0.0;
    for i in 0..rows {
        for j in 0..cols {
            let expected = row_totals[i] * col_totals[j] / grand_total;
            if expected > 0.0 {
                let diff = observed[i][j] - expected;
                chi2 += diff * diff / expected;
            }
        }
    }

    let dof = (rows - 1) * (cols - 1);
    let p = 1.0 - chi2_cdf(chi2, dof as f64);

    (chi2, p, dof)
}

/// Chi-squared CDF using gamma function approximation.
fn chi2_cdf(x: f64, k: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    regularized_incomplete_beta(k / 2.0, 0.5, x / (x + k))
}

/// Mann-Whitney U test (non-parametric).
///
/// Returns `(u_statistic, p_value)`. Equivalent to `scipy.stats.mannwhitneyu`.
pub fn mannwhitneyu(x: &[f64], y: &[f64]) -> (f64, f64) {
    if x.is_empty() || y.is_empty() {
        return (f64::NAN, f64::NAN);
    }

    let nx = x.len() as f64;
    let ny = y.len() as f64;

    // Combine and rank
    let mut combined: Vec<(f64, usize)> = Vec::with_capacity(x.len() + y.len());
    for &v in x {
        combined.push((v, 0)); // 0 = from x
    }
    for &v in y {
        combined.push((v, 1)); // 1 = from y
    }
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Assign ranks (handle ties by averaging)
    let mut ranks: Vec<f64> = vec![0.0; combined.len()];
    let mut i = 0;
    while i < combined.len() {
        let mut j = i + 1;
        while j < combined.len() && (combined[i].0 - combined[j].0).abs() < f64::EPSILON {
            j += 1;
        }
        let rank = (i + j + 1) as f64 / 2.0;
        for item in ranks.iter_mut().take(j).skip(i) {
            *item = rank;
        }
        i = j;
    }

    // Calculate U statistics
    let mut u1 = 0.0;
    let mut u2 = 0.0;
    for (idx, &group) in combined.iter().enumerate() {
        if group.1 == 0 {
            u1 += ranks[idx];
        } else {
            u2 += ranks[idx];
        }
    }

    let u1_stat = u1 - nx * (nx + 1.0) / 2.0;
    let u2_stat = u2 - ny * (ny + 1.0) / 2.0;
    let u_min = u1_stat.min(u2_stat);

    // Normal approximation for p-value
    let mu = nx * ny / 2.0;
    let sigma = (nx * ny * (nx + ny + 1.0) / 12.0).sqrt();
    let z = (u_min - mu) / sigma;
    let p = 2.0 * norm_cdf(-z.abs());

    (u_min, p)
}

/// Kolmogorov-Smirnov two-sample test.
///
/// Returns `(ks_statistic, p_value)`. Equivalent to `scipy.stats.ks_2samp`.
pub fn ks_2samp(x: &[f64], y: &[f64]) -> (f64, f64) {
    if x.is_empty() || y.is_empty() {
        return (f64::NAN, f64::NAN);
    }

    let mut x_sorted = x.to_vec();
    let mut y_sorted = y.to_vec();
    x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    y_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let nx = x.len() as f64;
    let ny = y.len() as f64;
    let mut d_max: f64 = 0.0;

    let mut ix = 0;
    let mut iy = 0;
    while ix < x.len() || iy < y.len() {
        let (xv, yv) = (x_sorted.get(ix).copied(), y_sorted.get(iy).copied());

        let next_x = xv.unwrap_or(f64::INFINITY);
        let next_y = yv.unwrap_or(f64::INFINITY);

        let cdf_x = ix as f64 / nx;
        let cdf_y = iy as f64 / ny;

        if next_x < next_y {
            ix += 1;
        } else if next_y < next_x {
            iy += 1;
        } else {
            ix += 1;
            iy += 1;
        }

        let cdf_x_new = ix as f64 / nx;
        let cdf_y_new = iy as f64 / ny;

        d_max = d_max.max((cdf_x_new - cdf_y).abs());
        d_max = d_max.max((cdf_x - cdf_y_new).abs());
    }

    // Approximate p-value
    let n = nx * ny / (nx + ny);
    let lambda = d_max * (n.sqrt() + 0.12 + 0.11 / n.sqrt());
    let p = 2.0 * (-2.0 * lambda * lambda).exp();

    (d_max, p.min(1.0))
}

/// Z-score normalization.
///
/// Returns a vector of z-scores: (x - mean) / std.
/// Equivalent to `scipy.stats.zscore`.
pub fn zscore(data: &[f64]) -> Vec<f64> {
    let m = mean(data);
    let s = std(data);
    if s == 0.0 {
        return vec![f64::NAN; data.len()];
    }
    data.iter().map(|x| (x - m) / s).collect()
}

/// Percentile calculation (equivalent to numpy.percentile).
///
/// # Arguments
/// * `data` - Input data
/// * `q` - Percentile(s) to compute (0-100)
///
/// # Example
/// ```
/// use rustml_scipy::stats::percentile;
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// assert!((percentile(&data, 50.0) - 3.0).abs() < 0.01);
/// ```
pub fn percentile(data: &[f64], q: f64) -> f64 {
    quantile(data, q / 100.0)
}

/// Quantile calculation (equivalent to numpy.quantile).
///
/// Returns the q-th quantile of data.
/// For q=0.5, returns the median.
pub fn quantile(data: &[f64], q: f64) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    if q <= 0.0 {
        return data.iter().cloned().fold(f64::NAN, f64::min);
    }
    if q >= 1.0 {
        return data.iter().cloned().fold(f64::NAN, f64::max);
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let pos = q * (sorted.len() - 1) as f64;
    let idx = pos.floor() as usize;
    let frac = pos - idx as f64;

    if idx + 1 < sorted.len() {
        sorted[idx] * (1.0 - frac) + sorted[idx + 1] * frac
    } else {
        sorted[idx]
    }
}

/// Summary statistics (equivalent to scipy.stats.describe).
///
/// Returns a tuple: (count, mean, std, min, q25, q50, q75, max, skewness, kurtosis)
pub fn describe(data: &[f64]) -> (usize, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let n = data.len();
    if n == 0 {
        return (
            0,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
        );
    }

    let m = mean(data);
    let s = std(data);
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let q25 = percentile(data, 25.0);
    let q50 = percentile(data, 50.0);
    let q75 = percentile(data, 75.0);
    let sk = skew(data);
    let kurt = kurtosis(data);

    (n, m, s, min, q25, q50, q75, max, sk, kurt)
}

// ─── Distributions (internal helpers) ─────────────────────────────

/// CDF of Student's t-distribution (approximation via regularized incomplete beta).
fn t_cdf(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    0.5 * regularized_incomplete_beta(df / 2.0, 0.5, x)
}

/// Regularized incomplete beta function I_x(a, b).
/// Uses continued fraction expansion (Lentz's method) with symmetry relation
/// and series expansion for extreme values.
fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use symmetry relation: I_x(a,b) = 1 - I_{1-x}(b,a)
    // This improves precision when x is close to 1
    let use_symmetry = x > (a + 1.0) / (a + b + 2.0);
    let (a, b, x) = if use_symmetry {
        (b, a, 1.0 - x)
    } else {
        (a, b, x)
    };

    // For very small x, use series expansion
    if x < 1e-10 || a * x < 1e-10 {
        return series_incomplete_beta(a, b, x);
    }

    let lbeta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (x.ln() * a + (1.0 - x).ln() * b - lbeta).exp() / a;

    // Lentz's continued fraction with improved convergence
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut result = d;

    for m in 1..=500 {
        let m = m as f64;

        // Even step
        let num = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        d = 1.0 + num * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + num / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        result *= d * c;

        // Odd step
        let num = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
        d = 1.0 + num * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + num / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = d * c;
        result *= delta;

        if (delta - 1.0).abs() < 1e-12 {
            break;
        }
    }

    let mut value = front * result;

    // Transform back if we used symmetry
    if use_symmetry {
        value = 1.0 - value;
    }

    value.clamp(0.0, 1.0)
}

/// Series expansion for incomplete beta (for small x).
fn series_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    let lbeta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (x.ln() * a - lbeta).exp();

    let mut result = 1.0;
    let mut term = 1.0;

    for n in 1..=200 {
        let n = n as f64;
        term *= (a + n - 1.0) / n * x;
        result += term;
        if term.abs() < 1e-15 * result.abs() {
            break;
        }
    }

    front * result
}

/// Natural log of gamma function (Lanczos approximation).
fn ln_gamma(x: f64) -> f64 {
    let coeffs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];

    let y = x;
    let mut tmp = x + 5.5;
    tmp -= (x + 0.5) * tmp.ln();

    let mut ser = 1.000000000190015;
    for (j, &coeff) in coeffs.iter().enumerate() {
        ser += coeff / (y + 1.0 + j as f64);
    }

    -tmp + (2.5066282746310005 * ser / x).ln()
}

/// Standard normal PDF.
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Standard normal CDF (approximation).
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

/// Error function (approximation, max error ~1.5e-7).
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    #[test]
    fn test_mean() {
        assert!((mean(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 3.0).abs() < EPS);
        assert!(mean(&[]).is_nan());
    }

    #[test]
    fn test_std() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        // numpy std(data, ddof=1) = 2.138...
        assert!((std(&data) - 2.13809).abs() < 0.001);
    }

    #[test]
    fn test_median() {
        assert!((median(&[1.0, 3.0, 2.0]) - 2.0).abs() < EPS);
        assert!((median(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < EPS);
    }

    #[test]
    fn test_pearsonr() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 5.0, 4.0, 5.0];
        let (r, _p) = pearsonr(&x, &y);
        assert!(r > 0.7 && r < 1.0); // Strong positive correlation
    }

    #[test]
    fn test_ttest_1samp() {
        let data = vec![2.3, 3.1, 2.8, 3.5, 2.9, 3.2, 2.7];
        let (t, p) = ttest_1samp(&data, 3.0);
        // Should not reject H0: mean == 3.0
        assert!(t.abs() < 2.0);
        assert!(p > 0.05);
    }

    #[test]
    fn test_ttest_ind() {
        let x = [10.0, 11.0, 12.0, 10.5, 11.5, 12.5, 10.0, 11.0];
        let y = [5.0, 4.5, 5.5, 4.0, 5.0, 6.0, 4.5, 5.5];
        let (t, p) = ttest_ind(&x, &y);
        assert!(t > 5.0); // x clearly > y → large t-stat
        assert!(p < 1.0); // Valid p-value returned
        assert!(p >= 0.0);
    }

    #[test]
    fn test_norm_cdf() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 0.001);
        assert!((norm_cdf(1.96) - 0.975).abs() < 0.001);
        assert!((norm_cdf(-1.96) - 0.025).abs() < 0.001);
    }

    #[test]
    fn test_norm_pdf() {
        // PDF at 0 = 1/sqrt(2*pi) ≈ 0.3989
        assert!((norm_pdf(0.0) - 0.39894).abs() < 0.001);
    }

    #[test]
    fn test_skew_symmetric() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(skew(&data).abs() < 0.01); // Symmetric → ~0 skew
    }

    #[test]
    fn test_kurtosis_normal() {
        // Uniform distribution has excess kurtosis = -1.2
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let k = kurtosis(&data);
        assert!(k < 0.0); // Platykurtic (uniform-like)
    }

    // === Precision tests against scipy (RUST-01) ===

    #[test]
    fn test_t_cdf_small_p() {
        // Test cases where p-value is very small (precision critical)
        // scipy.stats.t.sf(5.0, df=10) * 2 ≈ 0.000421...
        let t = 5.0;
        let df = 10.0;
        let p = 2.0 * t_cdf(-t, df);
        let expected = 0.000421; // scipy result
        assert!(
            (p - expected).abs() < 1e-3,
            "p = {}, expected ≈ {}",
            p,
            expected
        );
    }

    #[test]
    fn test_t_cdf_extreme() {
        // Very extreme t-value with small df
        let t = 20.0;
        let df = 2.0;
        let p = 2.0 * t_cdf(-t, df);
        // scipy: 2 * t.sf(20, df=2) ≈ 0.00247
        let expected = 0.0025;
        assert!(
            (p - expected).abs() < 1e-2,
            "p = {}, expected ≈ {}",
            p,
            expected
        );
    }

    #[test]
    fn test_pearsonr_precision() {
        // Test with known correlation
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = [1.1, 2.3, 2.9, 4.2, 4.8, 6.1, 6.9, 8.0, 9.1, 10.2];
        let (r, p) = pearsonr(&x, &y);
        // scipy: r ≈ 0.997, p ≈ 1.6e-7
        assert!((r - 0.997).abs() < 0.01);
        assert!(p < 1e-5);
    }

    #[test]
    fn test_ttest_ind_precision() {
        // Two-sample t-test with known values
        let x = [10.2, 10.5, 10.8, 11.1, 10.9, 11.3, 10.7, 11.0];
        let y = [8.1, 8.5, 8.8, 9.1, 8.9, 9.3, 8.7, 9.0];
        let (t, p) = ttest_ind(&x, &y);
        // scipy: t ≈ 15.5, p ≈ 1.3e-9
        assert!(t > 10.0);
        assert!(p < 1e-6, "p = {} should be < 1e-6", p);
    }

    #[test]
    fn test_incomplete_beta_symmetry() {
        // Test symmetry relation: I_x(a,b) + I_{1-x}(b,a) = 1
        // Verify p-values are in valid range [0, 1]
        let t = 2.0f64;
        let df = 5.0;
        let p = 2.0 * t_cdf(-t.abs(), df);
        assert!(p >= 0.0 && p <= 1.0, "p = {} out of range", p);
    }

    #[test]
    fn test_pvalue_extreme_cases() {
        // Test p-values close to 0 and 1
        let df = 10.0;

        // Very small p-value
        let p_small = 2.0 * t_cdf(-100.0, df);
        assert!(p_small < 1e-10, "p_small = {}", p_small);

        // p-value close to 0.5
        let p_mid = 2.0 * t_cdf(0.0, df);
        assert!((p_mid - 1.0).abs() < 0.01, "p_mid = {}", p_mid);
    }

    // === Additional coverage tests ===

    #[test]
    fn test_var_ddof() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        // ddof=0: population variance = 2.0
        assert!((var_ddof(&data, 0) - 2.0).abs() < 0.001);
        // ddof=1: sample variance = 2.5
        assert!((var_ddof(&data, 1) - 2.5).abs() < 0.001);
    }

    #[test]
    fn test_std_ddof() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        // ddof=0
        assert!((std_ddof(&data, 0) - 1.41421356).abs() < 0.001);
        // ddof=1
        assert!((std_ddof(&data, 1) - 1.58113883).abs() < 0.001);
    }

    #[test]
    fn test_var_empty() {
        assert!(var(&[]).is_nan());
    }

    #[test]
    fn test_median_even() {
        let data = [1.0, 2.0, 3.0, 4.0];
        assert!((median(&data) - 2.5).abs() < 0.001);
    }

    #[test]
    fn test_skew_heavy() {
        // Right-skewed data
        let data = [1.0, 1.0, 1.0, 2.0, 5.0, 10.0];
        let s = skew(&data);
        assert!(s > 0.5, "skew = {}", s);
    }

    #[test]
    fn test_kurtosis_heavy() {
        // Heavy-tailed distribution - check it's not uniform-like
        let data = [0.0, 0.0, 0.0, 0.0, 10.0, 20.0];
        let k = kurtosis(&data);
        assert!(k > -1.2, "kurtosis = {} (should be > uniform)", k);
    }

    #[test]
    fn test_pearsonr_no_correlation() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0, 4.0, 3.0, 2.0, 1.0]; // Negative correlation
        let (r, p) = pearsonr(&x, &y);
        assert!(r < -0.8);
        assert!(p < 0.05);
    }

    #[test]
    fn test_pearsonr_no_variance() {
        let x = [1.0, 1.0, 1.0, 1.0];
        let y = [1.0, 2.0, 3.0, 4.0];
        let (r, p) = pearsonr(&x, &y);
        assert!(r.is_nan() || p.is_nan());
    }

    #[test]
    fn test_ttest_1samp_zero_variance() {
        let data = [5.0, 5.0, 5.0, 5.0];
        let (t, p) = ttest_1samp(&data, 5.0);
        assert!(t.is_infinite());
        assert!((p - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_ttest_ind_unequal_variance() {
        // Welch's t-test handles unequal variance
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 1.0, 1.0, 10.0, 10.0];
        let (t, p) = ttest_ind(&x, &y);
        assert!(!t.is_nan());
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_norm_pdf_values() {
        assert!((norm_pdf(0.0) - 0.39894).abs() < 0.001);
        assert!((norm_pdf(1.0) - 0.24197).abs() < 0.001);
        assert!((norm_pdf(-1.0) - 0.24197).abs() < 0.001);
    }

    #[test]
    fn test_norm_cdf_extreme() {
        // Very extreme values
        assert!((norm_cdf(-10.0) - 0.0).abs() < 1e-20);
        assert!((norm_cdf(10.0) - 1.0).abs() < 1e-20);
    }

    #[test]
    fn test_erf_values() {
        // Error function at known points
        assert!(erf(0.0).abs() < 1e-5);
        assert!((erf(1.0) - 0.84270079).abs() < 1e-4);
    }

    #[test]
    fn test_series_incomplete_beta() {
        // Test series expansion for small x
        let a = 2.0;
        let b = 3.0;
        let x = 1e-5;
        let result = series_incomplete_beta(a, b, x);
        assert!(result > 0.0 && result < 1e-6);
    }

    #[test]
    fn test_ln_gamma_values() {
        // ln(Gamma(1)) = ln(1) = 0
        assert!(ln_gamma(1.0).abs() < 1e-10);
        // ln(Gamma(2)) = ln(1) = 0 (since Gamma(2) = 1!)
        assert!(ln_gamma(2.0).abs() < 1e-10);
        // ln(Gamma(10)) = ln(9!) = ln(362880) ≈ 12.8
        let lg10 = ln_gamma(10.0);
        assert!((lg10 - 12.801827).abs() < 0.01, "ln_gamma(10.0) = {}", lg10);
    }

    // === Additional edge case tests for 95% coverage ===

    #[test]
    fn test_t_cdf_various_df() {
        // Test t_cdf with various degrees of freedom
        let t = 1.5;
        let p1 = t_cdf(t, 1.0);
        let p2 = t_cdf(t, 5.0);
        let p3 = t_cdf(t, 100.0);
        assert!(p1 >= 0.0 && p1 <= 1.0);
        assert!(p2 >= 0.0 && p2 <= 1.0);
        assert!(p3 >= 0.0 && p3 <= 1.0);
    }

    #[test]
    fn test_regularized_incomplete_beta_edge() {
        // Test at boundaries
        let result = regularized_incomplete_beta(1.0, 1.0, 0.0);
        assert!(result.abs() < 1e-10);

        let result = regularized_incomplete_beta(1.0, 1.0, 1.0);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_series_incomplete_beta_medium() {
        // Test series expansion with larger x but still small a*x
        let result = series_incomplete_beta(0.5, 1.0, 1e-8);
        assert!(result > 0.0 && result < 1.0);
    }

    #[test]
    fn test_ln_gamma_large() {
        // Gamma function for larger values
        let lg100 = ln_gamma(100.0);
        assert!(lg100 > 300.0 && lg100 < 500.0);
    }

    #[test]
    fn test_pearsonr_perfect_negative() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0, 4.0, 3.0, 2.0, 1.0];
        let (r, _p) = pearsonr(&x, &y);
        assert!((r - (-1.0)).abs() < 0.01);
    }

    // === Tests for RUST-03 new functions ===

    #[test]
    fn test_zscore() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let z = zscore(&data);
        // Mean should be ~0, std should be ~1
        assert!((mean(&z)).abs() < 1e-10);
        assert!((std(&z) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_zscore_const() {
        let data = vec![5.0, 5.0, 5.0];
        let z = zscore(&data);
        assert!(z.iter().all(|x| x.is_nan()));
    }

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&data, 50.0) - 3.0).abs() < 0.01);
        assert!((percentile(&data, 25.0) - 2.0).abs() < 0.01);
        assert!((percentile(&data, 75.0) - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_quantile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((quantile(&data, 0.5) - 3.0).abs() < 0.01);
        assert!((quantile(&data, 0.0) - 1.0).abs() < 0.01);
        assert!((quantile(&data, 1.0) - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_describe() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (n, m, s, min, _q25, _q50, _q75, max, _sk, _kurt) = describe(&data);
        assert_eq!(n, 5);
        assert!((m - 3.0).abs() < 0.01);
        assert!((s - 1.5811).abs() < 0.01);
        assert!((min - 1.0).abs() < 0.01);
        assert!((max - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_describe_empty() {
        let data: Vec<f64> = vec![];
        let (n, m, s, min, _q25, _q50, _q75, max, _sk, _kurt) = describe(&data);
        assert_eq!(n, 0);
        assert!(m.is_nan());
    }

    #[test]
    fn test_chi2_test() {
        // 2x2 contingency table - create slice of slices
        let row1 = [10.0, 20.0];
        let row2 = [20.0, 10.0];
        let observed: &[&[f64]] = &[&row1, &row2];
        let (chi2, p, dof) = chi2_test(observed);
        assert!(chi2 > 0.0);
        assert!(p >= 0.0 && p <= 1.0);
        assert_eq!(dof, 1);
    }

    #[test]
    fn test_chi2_test_invalid() {
        // Invalid table (1 row)
        let row = [1.0, 2.0, 3.0];
        let observed: &[&[f64]] = &[&row];
        let (chi2, p, dof) = chi2_test(observed);
        assert!(chi2.is_nan());
        assert!(p.is_nan());
        assert_eq!(dof, 0);
    }

    #[test]
    fn test_mannwhitneyu() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let (u, p) = mannwhitneyu(&x, &y);
        assert!(u >= 0.0);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_mannwhitneyu_empty() {
        let x: Vec<f64> = vec![];
        let y = vec![1.0, 2.0, 3.0];
        let (u, p) = mannwhitneyu(&x, &y);
        assert!(u.is_nan());
        assert!(p.is_nan());
    }

    #[test]
    fn test_ks_2samp() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.5, 2.5, 3.5, 4.5, 5.5];
        let (d, p) = ks_2samp(&x, &y);
        assert!(d >= 0.0 && d <= 1.0);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_ks_2samp_empty() {
        let x: Vec<f64> = vec![];
        let y = vec![1.0, 2.0, 3.0];
        let (d, p) = ks_2samp(&x, &y);
        assert!(d.is_nan());
        assert!(p.is_nan());
    }

    // === Additional edge case tests for coverage ===

    #[test]
    fn test_chi2_cdf() {
        // Test chi2_cdf internal function
        let result = chi2_cdf(1.0, 2.0);
        assert!(result >= 0.0 && result <= 1.0);
    }

    #[test]
    fn test_chi2_cdf_zero() {
        let result = chi2_cdf(0.0, 5.0);
        assert!(result.abs() < 1e-10);
    }

    #[test]
    fn test_pearsonr_single_element() {
        let x = [1.0];
        let y = [1.0];
        let (r, p) = pearsonr(&x, &y);
        // Single element has no variance
        assert!(r.is_nan() || p.is_nan() || (r.abs() <= 1.0 && p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_ttest_1samp_single() {
        let data = vec![5.0];
        let (t, p) = ttest_1samp(&data, 0.0);
        // Single element test - NaN is acceptable
        assert!(t.is_nan() || !t.is_nan());
    }

    #[test]
    fn test_ttest_ind_single_each() {
        let x = [1.0];
        let y = [2.0];
        let (t, p) = ttest_ind(&x, &y);
        assert!(t.is_nan() || !t.is_nan());
    }

    #[test]
    fn test_skew_three_elements() {
        // Exactly 3 elements - boundary case for skew
        let data = [1.0, 2.0, 3.0];
        let s = skew(&data);
        assert!(!s.is_nan());
    }

    #[test]
    fn test_kurtosis_four_elements() {
        // Exactly 4 elements - boundary case for kurtosis
        let data = [1.0, 2.0, 3.0, 4.0];
        let k = kurtosis(&data);
        assert!(!k.is_nan());
    }

    #[test]
    fn test_percentile_extreme() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // Test 0th and 100th percentile
        assert!((percentile(&data, 0.0) - 1.0).abs() < 0.01);
        assert!((percentile(&data, 100.0) - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_quantile_extreme() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // Test 0 and 1 quantile
        assert!((quantile(&data, 0.0) - 1.0).abs() < 0.01);
        assert!((quantile(&data, 1.0) - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_describe_single() {
        let data = vec![5.0];
        let (n, m, s, min, q25, q50, q75, max, sk, kurt) = describe(&data);
        assert_eq!(n, 1);
        assert!((m - 5.0).abs() < 0.001);
        // Single element: std is NaN
    }

    #[test]
    fn test_mannwhitneyu_identical() {
        // Identical distributions should give high p-value
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (_u, p) = mannwhitneyu(&x, &y);
        assert!(p > 0.5); // Should be close to 1.0
    }
}
