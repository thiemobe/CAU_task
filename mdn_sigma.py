"""
mdn_sigma.py

Lookup-table version — no MDN calls during training.

Before training, load the CSV once with load_lookup_table().
During training, call get_sigma_from_lookup() instead of the MDN.

The CSV must have columns:
    state           - 'SET' or 'RESET'
    g_before_uS     - initial conductance in µS
    mixture_std_uS  - mixture standard deviation in µS  (this becomes sigma)
"""

import numpy as np
import pandas as pd


# ── low-voltage updates use the picked component's mean instead of sampling ───
LOW_VOLTAGE_MEAN_MIN_V = 0.1  # V
LOW_VOLTAGE_MEAN_MAX_V = 0.4  # V

# ── high-voltage pulses that stay on component 0 (no switch) use its mean too ─
HIGH_VOLTAGE_COMPONENT0_MEAN_MIN_V = 0.5  # V


# ── device constants ──────────────────────────────────────────────────────────
G_MIN_US = 5.0    # µS
G_MAX_US = 130.0  # µS
G_RANGE  = G_MAX_US - G_MIN_US  # 125 µS

# ── lookup-row matching defaults ──────────────────────────────────────────────
G_BEFORE_WEIGHT = 0.5
G_AFTER_WEIGHT = 0.5
MIN_PI_FOR_MEAN_MATCH = 0.2
MAX_RESAMPLE_ATTEMPTS = 50


def _weight_to_conductance_uS(w):
    """Convert normalized weight in [-1, 1] to conductance in µS."""
    g_uS = (float(w) + 1.0) / 2.0 * G_RANGE + G_MIN_US
    return float(np.clip(g_uS, G_MIN_US, G_MAX_US))


def _conductance_uS_to_weight(g_uS):
    """Convert conductance in µS to normalized weight in [-1, 1]."""
    w = 2.0 * (float(g_uS) - G_MIN_US) / G_RANGE - 1.0
    return float(np.clip(w, -1.0, 1.0))


def _desired_conductance_from_grad(w_current, grad):
    """
    Convert gradient to a desired next conductance target.

    We apply the usual gradient-descent step in weight space:
        w_desired = clip(w_current - grad, -1, 1)
    and then map w_desired linearly to conductance.
    """
    w_desired = float(np.clip(float(w_current) - float(grad), -1.0, 1.0))
    g_desired_uS = _weight_to_conductance_uS(w_desired)
    return g_desired_uS, w_desired


def _pick_best_row_by_g_before_and_desired_after(
    w_current,
    grad,
    df_set,
    df_reset,
    g_before_weight=G_BEFORE_WEIGHT,
    g_after_weight=G_AFTER_WEIGHT,
    min_pi_for_mean_match=MIN_PI_FOR_MEAN_MATCH,
    g_current_uS_override=None,
    voltage_range=None,
):
    """
    Pick best row using two conductance-space targets:
    1) current G_before (from current weight)
    2) desired G_after (from gradient-transformed desired weight)

    The match is component-aware and OR-based:
    - build candidates for (row, mu_1) and (row, mu_2) separately
    - keep only candidates whose corresponding pi >= min_pi_for_mean_match
    - score each candidate as:
        score = g_before_weight * d_before + g_after_weight * d_after_component
    - select the single best candidate (row + one component)

    Polarity is still selected from gradient sign to preserve the same
    SET/RESET direction semantics as the training loop.
    """
    if g_current_uS_override is None:
        g_current_uS = _weight_to_conductance_uS(w_current)
    else:
        g_current_uS = float(np.clip(float(g_current_uS_override), G_MIN_US, G_MAX_US))
    g_desired_uS, w_desired = _desired_conductance_from_grad(w_current, grad)

    polarity = 'RESET' if grad > 0 else 'SET'
    df = df_reset if polarity == 'RESET' else df_set
    if voltage_range is not None:
        if len(voltage_range) != 2:
            raise ValueError('voltage_range must contain a minimum and maximum voltage.')
        voltage_min, voltage_max = map(float, voltage_range)
        if not np.isfinite(voltage_min) or not np.isfinite(voltage_max) or voltage_min > voltage_max:
            raise ValueError('voltage_range must be finite and ordered from minimum to maximum.')
        if 'voltage' not in df.columns:
            raise ValueError('Lookup table must contain a voltage column for voltage filtering.')
        row_voltages = pd.to_numeric(df['voltage'], errors='coerce')
        df = df.loc[row_voltages.between(voltage_min, voltage_max, inclusive='both')]
        if df.empty:
            raise ValueError(
                f'No {polarity} lookup rows in voltage range '
                f'[{voltage_min:.4g}, {voltage_max:.4g}] V.'
            )
        df = df.reset_index(drop=True)

    g_before_vals = df['g_before_uS'].values.astype(float)
    mu1_vals = df['mu_1_uS'].values.astype(float)
    mu2_vals = df['mu_2_uS'].values.astype(float)
    pi1_vals = df['pi_1'].values.astype(float)
    pi2_vals = df['pi_2'].values.astype(float)

    d_before = np.abs(g_before_vals - g_current_uS) / G_RANGE
    d_after_mu1 = np.abs(mu1_vals - g_desired_uS) / G_RANGE
    d_after_mu2 = np.abs(mu2_vals - g_desired_uS) / G_RANGE

    # Component eligibility is strict: keep only components with pi >= threshold.
    mu1_eligible = np.isfinite(mu1_vals) & np.isfinite(pi1_vals) & (pi1_vals >= float(min_pi_for_mean_match))
    mu2_eligible = np.isfinite(mu2_vals) & np.isfinite(pi2_vals) & (pi2_vals >= float(min_pi_for_mean_match))

    score_mu1 = g_before_weight * d_before + g_after_weight * d_after_mu1
    score_mu2 = g_before_weight * d_before + g_after_weight * d_after_mu2
    score_mu1 = np.where(mu1_eligible, score_mu1, np.inf)
    score_mu2 = np.where(mu2_eligible, score_mu2, np.inf)

    best_idx_mu1 = int(np.argmin(score_mu1))
    best_idx_mu2 = int(np.argmin(score_mu2))
    best_score_mu1 = float(score_mu1[best_idx_mu1])
    best_score_mu2 = float(score_mu2[best_idx_mu2])

    if not np.isfinite(best_score_mu1) and not np.isfinite(best_score_mu2):
        raise ValueError(
            'No eligible lookup component found under pi threshold: '
            f'min_pi_for_mean_match={float(min_pi_for_mean_match):.3f}'
        )

    if best_score_mu1 <= best_score_mu2:
        best_idx = best_idx_mu1
        matched_component = 1
        best_score = best_score_mu1
        d_after_selected = float(d_after_mu1[best_idx])
    else:
        best_idx = best_idx_mu2
        matched_component = 2
        best_score = best_score_mu2
        d_after_selected = float(d_after_mu2[best_idx])

    best_row = df.iloc[best_idx]

    match_info = {
        'g_current_uS': float(g_current_uS),
        'g_desired_uS': float(g_desired_uS),
        'w_desired': float(w_desired),
        'd_before': float(d_before[best_idx]),
        'd_after': float(d_after_selected),
        'score': float(best_score),
        'preferred_mu_component': int(matched_component),
        'min_pi_for_mean_match': float(min_pi_for_mean_match),
        'threshold_relaxed': False,
    }
    return best_row, polarity, match_info


def load_lookup_table(csv_path):
    """
    Load the precomputed MDN predictions from CSV.
    Call this once before training and pass the result to get_sigma_from_lookup().

    Returns:
        df_set   - DataFrame filtered to SET rows only
        df_reset - DataFrame filtered to RESET rows only
    """
    df = pd.read_csv(csv_path)

    df_set   = df[df['state'] == 'SET'  ].reset_index(drop=True)
    df_reset = df[df['state'] == 'RESET'].reset_index(drop=True)

    print(f"Lookup table loaded: {len(df_set)} SET rows, {len(df_reset)} RESET rows")

    return df_set, df_reset


def get_sigma_from_lookup(w_current, grad, df_set, df_reset):
    """
    Given the current weight and gradient, find the best matching row
    in the lookup table and return its mixture_std_uS as sigma.

    Replaces the fixed sigma_mdn = 0.1 in the training loop.

    Parameters:
        w_current  - current weight (float, in [-1, 1])
        grad       - gradient value (float, in [-1, 1])
        df_set     - SET rows of the lookup table (from load_lookup_table)
        df_reset   - RESET rows of the lookup table (from load_lookup_table)

    Returns:
        sigma_mdn - noise std in weight space (dimensionless, for randn)
        sigma_uS       - raw sigma in µS (for logging)
        polarity       - 'SET' or 'RESET' (for logging)
    """

    best_row, polarity, _ = _pick_best_row_by_g_before_and_desired_after(
        w_current,
        grad,
        df_set,
        df_reset,
    )
    # ── Step 5: read sigma from that row ─────────────────────────────────────
    sigma_uS = best_row['mixture_std_uS']

    # ── Step 6: convert sigma from µS → weight space ─────────────────────────
    sigma_mdn = sigma_uS / G_RANGE
    #print(f"Converted σ: {sigma_uS:.2f}µS → {sigma_mdn:.4f} in weight space\n")

    return sigma_mdn, sigma_uS, polarity


def sample_weight_from_lookup(w_current, grad, df_set, df_reset, return_debug=False,
                               max_resample_attempts=MAX_RESAMPLE_ATTEMPTS, voltage_override=None,
                               post_pulse_read_noise=False, post_pulse_read_noise_std_uS=0.0,
                               enforce_monotonicity=True, voltage_range=None):
    """
    Sample a next weight from the best-match row's Gaussian mixture.

        The best row is chosen by matching:
            - row g_before_uS to current G_before
            - row (mu_1_uS, mu_2_uS) to desired G_after derived from gradient

        Then this function samples G_next from:
        p(G_next) = pi_1 * N(mu_1_uS, sigma_1_uS) + pi_2 * N(mu_2_uS, sigma_2_uS)

        Low-voltage exception:
        - if the best-match row's voltage falls within
            [LOW_VOLTAGE_MEAN_MIN_V, LOW_VOLTAGE_MEAN_MAX_V] (0.1-0.3V), G_next is
            set to the picked component's mean (mu) instead of a random draw from
            N(mu, sigma). The component is still picked the same way (weighted by
            pi), only the within-component sampling step is replaced.

        High-voltage component-0 exception:
        - if the best-match row's voltage is >= HIGH_VOLTAGE_COMPONENT0_MEAN_MIN_V
            (0.5V) and the picked component is component 0 (no switch occurred),
            G_next is set to that component's mean (p_max) instead of a random
            draw. Component 1 (an actual switch) still samples normally.

        Optional read-noise model:
        - if post_pulse_read_noise=True and post_pulse_read_noise_std_uS>0, a Gaussian read-noise
            sample is drawn in conductance space and added AFTER lookup matching and
            AFTER the sampled device-state update.
        - row matching always uses the clean device state.

        Optional monotonicity constraint:
        - if enforce_monotonicity=True (default), a SET pulse must not lower
            conductance and a RESET pulse must not raise conductance.
        - if enforce_monotonicity=False, this directional constraint is disabled.

    voltage_override is kept only for backward compatibility and is ignored in
    this row-selection mode.

    voltage_range optionally restricts selected lookup rows to an inclusive
    (minimum_voltage, maximum_voltage) interval. If no eligible row remains,
    the current weight is returned unchanged.

    Returns:
        w_sampled - sampled next weight in [-1, 1]
        g_sampled_uS - sampled next conductance in µS
        polarity - selected device polarity ('SET' or 'RESET')
        debug_info - optional dict with lookup/sampling details when return_debug=True
    """
    g_current_true_uS = _weight_to_conductance_uS(w_current)
    read_noise_uS = 0.0

    try:
        best_row, polarity, match_info = _pick_best_row_by_g_before_and_desired_after(
            w_current,
            grad,
            df_set,
            df_reset,
            g_current_uS_override=g_current_true_uS,
            voltage_range=voltage_range,
        )
    except ValueError:
        # No row/component matched the constraints; hold rather than pulse.
        if not return_debug:
            return float(w_current), float(g_current_true_uS), 'RESET' if grad > 0 else 'SET'

        debug_info = {
            'current_w': float(w_current),
            'enforce_monotonicity': bool(enforce_monotonicity),
            'use_component_mean': True,
            'post_pulse_read_noise': bool(post_pulse_read_noise),
            'post_pulse_read_noise_std_uS': float(post_pulse_read_noise_std_uS),
            'read_noise_uS': 0.0,
            'selected_voltage': float('nan'),
            'selected_polarity': 'RESET' if grad > 0 else 'SET',
            'g_current_uS': float(g_current_true_uS),
            'g_matching_uS': float(g_current_true_uS),
            'g_desired_uS': float('nan'),
            'w_desired': float('nan'),
            'match_d_before': float('nan'),
            'match_d_after': float('nan'),
            'match_score': float('nan'),
            'preferred_mu_component': 0,
            'min_pi_for_mean_match': float(MIN_PI_FOR_MEAN_MATCH),
            'mean_match_threshold_relaxed': False,
            'best_row_state': 'NONE',
            'best_row_voltage': float('nan'),
            'best_row_g_before_uS': float('nan'),
            'best_row_pi_1': float('nan'),
            'best_row_mu_1_uS': float('nan'),
            'best_row_sigma_1_uS': float('nan'),
            'best_row_pi_2': float('nan'),
            'best_row_mu_2_uS': float('nan'),
            'best_row_sigma_2_uS': float('nan'),
            'chosen_component_index': -1,
            'sampled_mu_uS': float('nan'),
            'sampled_sigma_uS': float('nan'),
            'sampled_g_uS': float(g_current_true_uS),
            'sampled_w': float(w_current),
            'pi_values': [1.0, 0.0],
            'mu_values': [float('nan'), float('nan')],
            'sigma_values': [float('nan'), float('nan')],
            'resample_attempts_used': 0,
            'constraint_enforced_by_clip': False,
            'no_eligible_component_under_pi_threshold': True,
        }
        return float(w_current), float(g_current_true_uS), ('RESET' if grad > 0 else 'SET'), debug_info

    # Component probabilities and parameters from CSV row.
    pi_values = np.array([best_row['pi_1'], best_row['pi_2']], dtype=float)
    mu_values = np.array([best_row['mu_1_uS'], best_row['mu_2_uS']], dtype=float)
    sigma_values = np.array([best_row['sigma_1_uS'], best_row['sigma_2_uS']], dtype=float)

    # Handle missing/invalid component 2 gracefully.
    valid = np.isfinite(pi_values) & np.isfinite(mu_values) & np.isfinite(sigma_values)
    pi_values = np.where(valid, np.maximum(pi_values, 0.0), 0.0)

    # Fallback to component 1 if probabilities are degenerate.
    pi_sum = np.sum(pi_values)
    if pi_sum <= 0:
        pi_values_normalized = np.zeros_like(pi_values)
        pi_values_normalized[0] = 1.0
    else:
        pi_values_normalized = pi_values / pi_sum

    row_voltage = float(best_row['voltage']) if 'voltage' in best_row.index else float('nan')
    use_component_mean = np.isfinite(row_voltage) and (
        LOW_VOLTAGE_MEAN_MIN_V <= row_voltage <= LOW_VOLTAGE_MEAN_MAX_V
    )
    use_component0_mean_if_no_switch = (
        np.isfinite(row_voltage) and row_voltage >= HIGH_VOLTAGE_COMPONENT0_MEAN_MIN_V
    )

    def _draw_once():
        comp_idx = int(np.random.choice(len(pi_values_normalized), p=pi_values_normalized))
        mu = float(mu_values[comp_idx])
        sigma = float(max(sigma_values[comp_idx], 1e-9))
        used_mean = use_component_mean or (use_component0_mean_if_no_switch and comp_idx == 0)
        if used_mean:
            g = float(np.clip(mu, G_MIN_US, G_MAX_US))
        else:
            g = float(np.clip(np.random.normal(loc=mu, scale=sigma), G_MIN_US, G_MAX_US))
        return comp_idx, mu, sigma, g, used_mean

    def _is_compliant(g):
        if polarity == 'SET':
            return g >= g_current_true_uS
        return g <= g_current_true_uS  # RESET

    component_index, mu_sel, sigma_sel, g_sampled_uS, used_mean = _draw_once()
    attempts_used = 1
    constraint_enforced_by_clip = False
    if enforce_monotonicity:
        while not _is_compliant(g_sampled_uS) and attempts_used < max_resample_attempts:
            component_index, mu_sel, sigma_sel, g_sampled_uS, used_mean = _draw_once()
            attempts_used += 1

        if not _is_compliant(g_sampled_uS):
            # Gave up resampling; hold at current conductance (no effective change)
            # rather than violate the device's physical switching direction.
            g_sampled_uS = g_current_true_uS
            constraint_enforced_by_clip = True

    # Read noise is modeled as a post-update perturbation of the observed/applied
    # conductance, not as part of the row-matching state.
    if post_pulse_read_noise and post_pulse_read_noise_std_uS > 0:
        read_noise_uS = float(np.random.normal(loc=0.0, scale=float(post_pulse_read_noise_std_uS)))
        g_sampled_uS = float(np.clip(g_sampled_uS + read_noise_uS, G_MIN_US, G_MAX_US))

    # Convert sampled conductance back to weight space [-1, 1].
    w_sampled = _conductance_uS_to_weight(g_sampled_uS)

    if not return_debug:
        return w_sampled, float(g_sampled_uS), polarity

    debug_info = {
        'current_w': float(w_current),
        'enforce_monotonicity': bool(enforce_monotonicity),
        'use_component_mean': bool(used_mean),
        'post_pulse_read_noise': bool(post_pulse_read_noise),
        'post_pulse_read_noise_std_uS': float(post_pulse_read_noise_std_uS),
        'read_noise_uS': float(read_noise_uS),
        'selected_voltage': float(best_row['voltage']) if 'voltage' in best_row.index else float('nan'),
        'selected_polarity': polarity,
        'g_current_uS': float(g_current_true_uS),
        'g_matching_uS': float(g_current_true_uS),
        'g_desired_uS': float(match_info['g_desired_uS']),
        'w_desired': float(match_info['w_desired']),
        'match_d_before': float(match_info['d_before']),
        'match_d_after': float(match_info['d_after']),
        'match_score': float(match_info['score']),
        'preferred_mu_component': int(match_info['preferred_mu_component']),
        'min_pi_for_mean_match': float(match_info['min_pi_for_mean_match']),
        'mean_match_threshold_relaxed': bool(match_info['threshold_relaxed']),
        'best_row_state': str(best_row['state']),
        'best_row_voltage': float(best_row['voltage']),
        'best_row_g_before_uS': float(best_row['g_before_uS']),
        'best_row_pi_1': float(best_row['pi_1']),
        'best_row_mu_1_uS': float(best_row['mu_1_uS']),
        'best_row_sigma_1_uS': float(best_row['sigma_1_uS']),
        'best_row_pi_2': float(best_row['pi_2']),
        'best_row_mu_2_uS': float(best_row['mu_2_uS']),
        'best_row_sigma_2_uS': float(best_row['sigma_2_uS']),
        'chosen_component_index': int(component_index),
        'sampled_mu_uS': mu_sel,
        'sampled_sigma_uS': sigma_sel,
        'sampled_g_uS': float(g_sampled_uS),
        'sampled_w': float(w_sampled),
        'pi_values': pi_values_normalized.tolist(),
        'mu_values': mu_values.tolist(),
        'sigma_values': sigma_values.tolist(),
        'resample_attempts_used': attempts_used,
        'constraint_enforced_by_clip': constraint_enforced_by_clip,
    }

    return w_sampled, float(g_sampled_uS), polarity, debug_info