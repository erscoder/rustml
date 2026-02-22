//! # rustml-scipy optimize
//!
//! Optimization algorithms ported from scipy.optimize.

/// Brent's method for 1D minimization
pub fn minimize_scalar<F>(f: F, a: f64, b: f64) -> Result<(f64, f64), String>
where
    F: Fn(f64) -> f64,
{
    const GOLDEN_RATIO: f64 = 1.618033988749895;
    const TOL: f64 = 1e-8;
    const MAX_ITER: usize = 100;

    if a >= b {
        return Err("a must be less than b".to_string());
    }

    let mut xa = a;
    let mut xb = b;
    let mut xc = xb - (xb - xa) / GOLDEN_RATIO;
    let mut xd = xa + (xb - xa) / GOLDEN_RATIO;

    let mut fc = f(xc);
    let mut fd = f(xd);

    for _ in 0..MAX_ITER {
        if fc < fd {
            xb = xd;
            xd = xc;
            fd = fc;
            xc = xb - (xb - xa) / GOLDEN_RATIO;
            fc = f(xc);
        } else {
            xa = xc;
            xc = xd;
            fc = fd;
            xd = xa + (xb - xa) / GOLDEN_RATIO;
            fd = f(xd);
        }

        if (xb - xa).abs() < TOL {
            break;
        }
    }

    if fc < fd {
        Ok((xc, fc))
    } else {
        Ok((xd, fd))
    }
}

/// Bisection method for root finding
pub fn root_scalar<F>(f: F, a: f64, b: f64) -> Result<f64, String>
where
    F: Fn(f64) -> f64,
{
    const TOL: f64 = 1e-8;
    const MAX_ITER: usize = 100;

    let fa = f(a);
    let fb = f(b);

    if fa * fb > 0.0 {
        return Err("Function must have opposite signs at a and b".to_string());
    }

    let mut xa = a;
    let mut xb = b;

    for _ in 0..MAX_ITER {
        let xc = (xa + xb) / 2.0;
        let fc = f(xc);

        if fc.abs() < TOL {
            return Ok(xc);
        }

        if fc * fa < 0.0 {
            xb = xc;
        } else {
            xa = xc;
        }

        if (xb - xa).abs() < TOL {
            return Ok(xc);
        }
    }

    Err("Max iterations reached".to_string())
}

/// Newton-Raphson method for root finding
pub fn newton<F, DF>(f: F, df: DF, x0: f64) -> Result<f64, String>
where
    F: Fn(f64) -> f64,
    DF: Fn(f64) -> f64,
{
    const TOL: f64 = 1e-8;
    const MAX_ITER: usize = 50;

    let mut x = x0;

    for _ in 0..MAX_ITER {
        let fx = f(x);
        let dfx = df(x);

        if dfx.abs() < TOL {
            return Err("Derivative is zero".to_string());
        }

        let x_new = x - fx / dfx;

        if (x_new - x).abs() < TOL {
            return Ok(x_new);
        }

        x = x_new;
    }

    Err("Max iterations reached".to_string())
}

/// Nelder-Mead simplex method for multivariate minimization
pub fn minimize<F>(f: F, x0: &[f64]) -> Result<Vec<f64>, String>
where
    F: Fn(&[f64]) -> f64,
{
    const TOL: f64 = 1e-8;
    const MAX_ITER: usize = 200;
    const ALPHA: f64 = 1.0;
    const GAMMA: f64 = 2.0;
    const RHO: f64 = 0.5;
    const SIGMA: f64 = 0.5;

    let n = x0.len();
    if n < 2 {
        return Err("Need at least 2 dimensions".to_string());
    }

    let mut simplex: Vec<Vec<f64>> = vec![x0.to_vec()];

    for i in 0..n {
        let mut point = x0.to_vec();
        if i == 0 {
            point[0] += 0.5;
        } else {
            point[i] += 0.5;
        }
        simplex.push(point);
    }

    let mut values: Vec<f64> = simplex.iter().map(|x| f(x)).collect();

    for _ in 0..MAX_ITER {
        let (worst_idx, _) = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let (best_idx, _) = values
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let mut centroid = vec![0.0; n];
        for (i, point) in simplex.iter().enumerate() {
            if i != worst_idx {
                for j in 0..n {
                    centroid[j] += point[j];
                }
            }
        }
        #[allow(clippy::needless_range_loop)]
        for j in 0..n {
            centroid[j] /= n as f64;
        }

        let range = values
            .iter()
            .fold(0.0_f64, |acc, &x| acc.max(x) - acc.min(x));
        if range < TOL {
            return Ok(simplex[best_idx].clone());
        }

        // Reflection
        let mut xr = centroid.clone();
        for (j, val) in xr.iter_mut().take(n).enumerate() {
            *val = centroid[j] + ALPHA * (centroid[j] - simplex[worst_idx][j]);
        }
        let fxr = f(&xr);

        if fxr < values[best_idx] {
            let mut xe = centroid.clone();
            for j in 0..n {
                xe[j] = centroid[j] + GAMMA * (xr[j] - centroid[j]);
            }
            let fxe = f(&xe);

            if fxe < fxr {
                simplex[worst_idx] = xe;
                values[worst_idx] = fxe;
            } else {
                simplex[worst_idx] = xr;
                values[worst_idx] = fxr;
            }
        } else {
            let mut sorted: Vec<_> = values.iter().enumerate().collect();
            sorted.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
            let second_worst_idx = sorted[1].0;

            if fxr < values[second_worst_idx] {
                simplex[worst_idx] = xr;
                values[worst_idx] = fxr;
            } else {
                let mut xc = centroid.clone();
                for j in 0..n {
                    xc[j] = centroid[j] + RHO * (simplex[worst_idx][j] - centroid[j]);
                }
                let fxc = f(&xc);

                if fxc < values[worst_idx] {
                    simplex[worst_idx] = xc;
                    values[worst_idx] = fxc;
                } else {
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..simplex.len() {
                        if i != best_idx {
                            for j in 0..n {
                                simplex[i][j] = simplex[best_idx][j]
                                    + SIGMA * (simplex[i][j] - simplex[best_idx][j]);
                            }
                            values[i] = f(&simplex[i]);
                        }
                    }
                }
            }
        }
    }

    let (best_idx, _) = values
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    Ok(simplex[best_idx].clone())
}

/// Curve fitting using least squares.
///
/// Fits a function `f(x, params)` to data `(x_data, y_data)`.
/// Returns the optimal parameters.
///
/// # Arguments
/// * `f` - The model function `f(x, params) -> f64`
/// * `x_data` - Input x values
/// * `y_data` - Observed y values
/// * `p0` - Initial guess for parameters
///
/// # Example
/// ```
/// use rustml_scipy::optimize::curve_fit;
/// let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
/// let y = vec![1.0, 2.9, 4.1, 5.0, 5.9]; // y = a*x + b
/// let f = |x: f64, params: &[f64]| params[0] * x + params[1];
/// let result = curve_fit(f, &x, &y, &[1.0, 0.0]).unwrap();
/// assert!((result[0] - 1.2).abs() < 0.1); // slope ~1.2
/// assert!((result[1] - 1.0).abs() < 0.5); // intercept ~1
/// ```
pub fn curve_fit<F>(f: F, x_data: &[f64], y_data: &[f64], p0: &[f64]) -> Result<Vec<f64>, String>
where
    F: Fn(f64, &[f64]) -> f64,
{
    if x_data.len() != y_data.len() {
        return Err("x_data and y_data must have same length".to_string());
    }
    if x_data.is_empty() {
        return Err("Empty data".to_string());
    }

    // Cost function: sum of squared residuals
    let cost = |params: &[f64]| -> f64 {
        let mut sum = 0.0;
        for (x, y) in x_data.iter().zip(y_data.iter()) {
            let y_pred = f(*x, params);
            let resid = y_pred - y;
            sum += resid * resid;
        }
        sum
    };

    // Use Nelder-Mead to minimize
    let result = minimize(cost, p0)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimize_scalar() {
        let f = |x: f64| (x - 3.0).powi(2);
        let (x_min, f_min) = minimize_scalar(f, 0.0, 6.0).unwrap();
        assert!((x_min - 3.0).abs() < 0.01);
        assert!(f_min.abs() < 0.01);
    }

    #[test]
    fn test_minimize_scalar_sin() {
        let f = |x: f64| x.sin();
        let (x_min, f_min) = minimize_scalar(f, 0.0, 10.0).unwrap();
        assert!((x_min - 4.71).abs() < 0.1);
        assert!((f_min - (-1.0)).abs() < 0.1);
    }

    #[test]
    fn test_root_scalar() {
        let f = |x: f64| x * x - 4.0;
        let root = root_scalar(f, 0.0, 4.0).unwrap();
        assert!((root - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_root_scalar_cubic() {
        let f = |x: f64| x.powi(3) - 8.0;
        let root = root_scalar(f, 0.0, 4.0).unwrap();
        assert!((root - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_newton() {
        let f = |x: f64| x * x - 4.0;
        let df = |x: f64| 2.0 * x;
        let root = newton(f, df, 10.0).unwrap();
        assert!((root - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_minimize_2d() {
        let f = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] - 4.0).powi(2);
        let result = minimize(f, &[0.0, 0.0]).unwrap();
        assert!((result[0] - 3.0).abs() < 0.1);
        assert!((result[1] - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_curve_fit_linear() {
        // Fit y = a*x + b to data
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let model = |x: f64, params: &[f64]| params[0] * x + params[1];
        let result = curve_fit(model, &x, &y, &[1.0, 0.0]).unwrap();
        // Should fit y = 2*x + 1
        assert!((result[0] - 2.0).abs() < 0.2);
        assert!((result[1] - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_curve_fit_quadratic() {
        // Fit y = a*x^2 + b to data (no linear term)
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0]; // y = x^2
        let model = |x: f64, params: &[f64]| params[0] * x * x + params[1];
        let result = curve_fit(model, &x, &y, &[1.0, 0.0]).unwrap();
        assert!((result[0] - 1.0).abs() < 0.1);
    }
}
