"""Simulate straddle implied vol curves under Student-t, normal, and lognormal terminal distributions."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from black_scholes import implied_vol_from_straddle


def student_t_scale(std: float, df: float) -> float:
    if df <= 2:
        raise ValueError("Student-t variance is finite only when df > 2.")
    if std <= 0:
        raise ValueError("Target standard deviation must be positive.")

    variance_factor = df / (df - 2)
    return std / np.sqrt(variance_factor)


def lognormal_parameters(mean: float, std: float) -> tuple[float, float]:
    if mean <= 0:
        raise ValueError("Lognormal mean must be positive.")
    if std <= 0:
        raise ValueError("Lognormal standard deviation must be positive.")

    variance_ratio = (std / mean) ** 2
    sigma_sq = np.log1p(variance_ratio)
    mu = np.log(mean) - 0.5 * sigma_sq
    sigma = np.sqrt(sigma_sq)
    return mu, sigma


def compute_straddle_curve(
    label: str,
    terminal_prices: np.ndarray,
    strikes: np.ndarray,
    discount_factor: float,
    spot: float,
    rate: float,
    time_to_maturity: float,
) -> list[dict[str, float | str]]:
    records = []
    for strike in strikes:
        payoff = np.abs(terminal_prices - strike)
        price = discount_factor * payoff.mean()
        implied_vol = implied_vol_from_straddle(price, spot, strike, rate, time_to_maturity)
        records.append({
            "distribution": label,
            "strike": strike,
            "price": price,
            "implied_vol": implied_vol,
        })
    return records


def summarize_distribution(label: str, terminal_prices: np.ndarray) -> dict[str, float | str]:
    series = pd.Series(terminal_prices)
    return {
        "distribution": label,
        "mean": series.mean(),
        "std": series.std(),
        "skew": series.skew(),
        "kurtosis": series.kurtosis(),
        "min": series.min(),
        "max": series.max(),
    }


def main(
    dfs: tuple[float, ...] = (10.0, 30.0),
    include_normal: bool = True,
    include_lognormal: bool = True,
    n_paths: int = 10**7,
    mean_terminal: float = 100.0,
    std_terminal: float = 10.0,
    strike_start: float = 80.0,
    strike_end: float = 120.0,
    strike_step: float = 5.0,
    spot: float = 100.0,
    rate: float = 0.0,
    time_to_maturity: float = 1.0,
    plot_terminal_distributions: bool = False,
    seed: int = 12345,
) -> None:
    if any(df <= 2 for df in dfs):
        raise ValueError("All degrees of freedom must exceed 2 for finite variance.")

    strikes = np.arange(strike_start, strike_end + strike_step, strike_step)
    rng = np.random.default_rng(seed)
    discount_factor = np.exp(-rate * time_to_maturity)

    params = [
        ("dfs", ", ".join(f"{df:g}" for df in dfs)),
        ("include_normal", include_normal),
        ("include_lognormal", include_lognormal),
        ("n_paths", n_paths),
        ("mean_terminal", mean_terminal),
        ("std_terminal", std_terminal),
        ("spot", spot),
        ("rate", rate),
        ("time_to_maturity", time_to_maturity),
        ("strike_start", strike_start),
        ("strike_end", strike_end),
        ("strike_step", strike_step),
        ("plot_terminal_distributions", plot_terminal_distributions),
        ("seed", seed),
    ]
    params_df = pd.DataFrame(params, columns=["parameter", "value"])
    print("Simulation parameters:")
    print(params_df.to_string(index=False))

    curve_records: list[dict[str, float | str]] = []
    summary_records: list[dict[str, float | str]] = []
    distribution_samples: dict[str, np.ndarray] = {}

    for df in dfs:
        scale = student_t_scale(std_terminal, df)
        terminal_prices = stats.t.rvs(df, loc=mean_terminal, scale=scale, size=n_paths, random_state=rng)
        terminal_prices = np.clip(terminal_prices, a_min=0.0, a_max=None)
        label = f"Student-t df={df:g}"
        distribution_samples[label] = terminal_prices
        summary_records.append(summarize_distribution(label, terminal_prices))
        curve_records.extend(
            compute_straddle_curve(label, terminal_prices, strikes, discount_factor, spot, rate, time_to_maturity)
        )

    if include_normal:
        terminal_prices = rng.normal(loc=mean_terminal, scale=std_terminal, size=n_paths)
        terminal_prices = np.clip(terminal_prices, a_min=0.0, a_max=None)
        label = "Normal"
        distribution_samples[label] = terminal_prices
        summary_records.append(summarize_distribution(label, terminal_prices))
        curve_records.extend(
            compute_straddle_curve(label, terminal_prices, strikes, discount_factor, spot, rate, time_to_maturity)
        )

    if include_lognormal:
        mu, sigma = lognormal_parameters(mean_terminal, std_terminal)
        terminal_prices = rng.lognormal(mean=mu, sigma=sigma, size=n_paths)
        label = "Lognormal"
        distribution_samples[label] = terminal_prices
        summary_records.append(summarize_distribution(label, terminal_prices))
        curve_records.extend(
            compute_straddle_curve(label, terminal_prices, strikes, discount_factor, spot, rate, time_to_maturity)
        )

    summary_df = pd.DataFrame.from_records(summary_records)
    print("Distribution summary:")
    print(summary_df.to_string(index=False))

    curve_df = pd.DataFrame.from_records(curve_records)
    print("\nStraddle results:")
    print(curve_df.to_string(index=False))

    plt.figure(figsize=(9, 5))
    for label, group in curve_df.groupby("distribution"):
        plt.plot(group["strike"], group["implied_vol"], marker="o", linestyle="-", label=label)

    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.title("Straddle Implied Volatility vs Strike (Normal, Lognormal, and Student-t)")
    plt.grid(True)
    plt.legend(title="Distribution")
    plt.tight_layout()

    if plot_terminal_distributions:
        plt.figure(figsize=(9, 5))
        x_min = min(samples.min() for samples in distribution_samples.values())
        x_max = max(samples.max() for samples in distribution_samples.values())
        if x_max <= x_min:
            x_max = x_min + 1.0
        grid = np.linspace(x_min, x_max, 400)
        for label, samples in distribution_samples.items():
            kde = stats.gaussian_kde(samples)
            plt.plot(grid, kde(grid), linewidth=1.5, label=label)

        plt.xlabel("Terminal Price")
        plt.ylabel("Density")
        plt.title("Terminal Price Distributions")
        plt.grid(True)
        plt.legend(title="Distribution")
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
