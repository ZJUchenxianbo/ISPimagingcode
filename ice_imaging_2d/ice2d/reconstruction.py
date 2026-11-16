"""Bounded regularized Gauss--Newton using discrete interface sensitivities."""

from pathlib import Path
import numpy as np
from .sensitivity import linearize, normal_system, sensitivity_diagnostics, ForwardBudgetExceeded


def regularizer(inverse):
    n = len(inverse.control_x) + 1
    bounds = np.array([inverse.top_bound] + [inverse.bottom_bound] * (n - 1))
    matrix = inverse.regularization * np.diag(1 / bounds ** 2)
    # Slope jumps of the bottom controls, nondimensionalized with node span.
    x = np.asarray(inverse.control_x)
    for j in range(1, n - 2):
        row = np.zeros(n)
        row[j] = 1 / (x[j] - x[j - 1])
        row[j + 1] = -1 / (x[j] - x[j - 1]) - 1 / (x[j + 1] - x[j])
        row[j + 2] = 1 / (x[j + 1] - x[j])
        row *= (x[-1] - x[0]) / inverse.bottom_bound
        matrix += inverse.smoothness * np.outer(row, row)
    return matrix, bounds


def reconstruct(forward, objective, output, mode="joint", strategy="full", model_budget=16):
    inverse = forward.inverse
    n = forward.parameter_count
    active = np.array([0]) if mode == "top" else np.arange(1, n) if mode == "bottom" else np.arange(n)
    theta = np.zeros(n)
    prior, bounds = regularizer(inverse)
    history = []
    predicted = objective.reference
    residual = objective.residual(predicted)
    initial_columns = linearize(forward, objective, theta, Path(output) / "cache/initial")
    initial_diagnostics = {name: sensitivity_diagnostics(initial_columns, residual, objective, name)
                           for name in ("bottom", "internal", "full")}
    frozen_columns = initial_columns
    forward.budget = forward.calls + max(0, model_budget - 2 - 2 * n) if strategy == "frozen" else None
    status = "iteration_limit"
    for iteration in range(min(inverse.max_iterations, 2) if strategy == "frozen" else inverse.max_iterations):
        norm_before = objective.norm(residual)
        before = 0.5 * (norm_before ** 2 + theta @ prior @ theta)
        if norm_before < inverse.relative_tolerance:
            status = "residual_tolerance"
            break
        columns = frozen_columns[active] if strategy == "frozen" else linearize(forward, objective, theta, Path(output) / "cache/iteration", active)
        gram, rhs = normal_system(columns, residual, objective)
        del columns
        if np.any(np.diag(gram) <= 1e-20):
            status = "unobservable_active_parameter"
            break
        system = gram + prior[np.ix_(active, active)]
        step_active = np.linalg.lstsq(system, rhs - (prior @ theta)[active], rcond=1e-10)[0]
        step = np.zeros(n)
        step[active] = step_active
        step *= min(1.0, inverse.max_step / max(np.max(abs(step)), 1e-30))
        accepted, used_step = False, 0.0
        for backtrack in range(inverse.max_backtracks):
            fraction = 0.5 ** backtrack
            candidate = np.clip(theta + fraction * step, -bounds, bounds)
            try:
                forward.geometry(candidate)
                # Accepted iterates must also admit the next central-difference
                # stencil, including the full final identifiability diagnostic.
                for parameter in range(n):
                    for sign in (-1, 1):
                        perturbed = candidate.copy()
                        perturbed[parameter] += sign * inverse.difference_step
                        forward.geometry(perturbed)
            except ValueError:
                continue
            try:
                trial = forward.evaluate(candidate)
            except ForwardBudgetExceeded:
                status = "forward_budget_exhausted"
                break
            trial_residual = objective.residual(trial)
            trial_norm = objective.norm(trial_residual)
            value = 0.5 * (trial_norm ** 2 + candidate @ prior @ candidate)
            if value < before and trial_norm < norm_before:
                theta, predicted, residual = candidate, trial, trial_residual
                accepted, used_step = True, fraction
                break
        after_norm = objective.norm(residual)
        history.append(dict(iteration=iteration, residual_before=norm_before, residual_after=after_norm,
                            objective_before=float(before),
                            objective_after=float(0.5 * (after_norm ** 2 + theta @ prior @ theta)),
                            accepted=accepted, step_fraction=used_step,
                            top_offset=float(theta[0]), maximum_bottom_offset=float(np.max(abs(theta[1:])))))
        print(f"{mode} iteration {iteration + 1}: residual {norm_before:.5g} -> {after_norm:.5g}", flush=True)
        if not accepted:
            status = "forward_budget_exhausted" if status == "forward_budget_exhausted" else "no_descent_step"
            break
        if norm_before - after_norm < inverse.relative_tolerance * max(norm_before, 1e-20):
            status = "relative_improvement_tolerance"
            break
    if strategy == "frozen":
        final_diagnostics = initial_diagnostics
    else:
        columns = linearize(forward, objective, theta, Path(output) / "cache/final")
        final_diagnostics = {name: sensitivity_diagnostics(columns, residual, objective, name)
                             for name in ("bottom", "internal", "full")}
        del columns
    return dict(theta=theta, geometry=forward.geometry(theta), predicted=predicted,
                history=history, status=status, initial_diagnostics=initial_diagnostics,
                final_diagnostics=final_diagnostics, initial_norm=objective.norm(objective.initial_residual),
                final_norm=objective.norm(residual))
