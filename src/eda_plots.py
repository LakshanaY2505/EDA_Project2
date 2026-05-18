"""
eda_plots.py
------------
EDA visualisation functions for the Telco churn dataset.
All functions accept a cleaned DataFrame and save a PNG to `save_dir`.

Usage:
    from src.eda_plots import plot_churn_rate, plot_contract_vs_churn, ...
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# ── Global style ───────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
TEAL   = "#5DCAA5"
ORANGE = "#D85A30"
AMBER  = "#FAC775"


def _save(fig: plt.Figure, name: str, save_dir: str) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = Path(save_dir) / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


def plot_churn_rate(df: pd.DataFrame, save_dir: str = "images") -> None:
    """Bar + pie showing overall churn distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    counts = df["Churn"].value_counts()
    axes[0].bar(["Stayed (0)", "Churned (1)"], counts.values,
                color=[TEAL, ORANGE], edgecolor="white")
    axes[0].set_title("Churn counts", fontweight="bold")
    axes[0].set_ylabel("Customers")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 30, str(v), ha="center", fontweight="bold")
    sns.despine(ax=axes[0])

    props = df["Churn"].value_counts(normalize=True)
    axes[1].pie(props.values, labels=["Stayed", "Churned"],
                autopct="%1.1f%%", colors=[TEAL, ORANGE], startangle=90)
    axes[1].set_title("Churn proportion", fontweight="bold")

    fig.tight_layout()
    _save(fig, "churn_rate.png", save_dir)


def plot_contract_vs_churn(df: pd.DataFrame, save_dir: str = "images") -> None:
    """Bar chart: churn rate per contract type."""
    data = (df.groupby("Contract")["Churn"].mean() * 100
              .sort_values(ascending=False).reset_index())
    data.columns = ["Contract", "Churn Rate"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(data["Contract"], data["Churn Rate"],
                  color=[ORANGE, AMBER, TEAL], edgecolor="white", width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", fontweight="bold", fontsize=11)
    ax.set_title("Churn rate by contract type", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Churn rate (%)")
    ax.set_ylim(0, 55)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    sns.despine(ax=ax)
    fig.tight_layout()
    _save(fig, "contract_vs_churn.png", save_dir)


def plot_tenure_distribution(df: pd.DataFrame, save_dir: str = "images") -> None:
    """Overlapping histograms + median comparison for tenure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for label, color, group in [("Stayed", TEAL, 0), ("Churned", ORANGE, 1)]:
        subset = df[df["Churn"] == group]["tenure"]
        axes[0].hist(subset, bins=25, alpha=0.6, color=color,
                     label=f"{label} (n={len(subset):,})", edgecolor="white")
    axes[0].set_title("Tenure distribution by churn", fontweight="bold")
    axes[0].set_xlabel("Tenure (months)")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    sns.despine(ax=axes[0])

    medians = df.groupby("Churn")["tenure"].median()
    axes[1].bar(["Stayed", "Churned"], medians.values,
                color=[TEAL, ORANGE], edgecolor="white", width=0.4)
    for i, v in enumerate(medians.values):
        axes[1].text(i, v + 0.5, f"{v:.0f} mo", ha="center", fontweight="bold")
    axes[1].set_title("Median tenure by churn", fontweight="bold")
    axes[1].set_ylabel("Months")
    axes[1].set_ylim(0, medians.max() * 1.25)
    sns.despine(ax=axes[1])

    fig.tight_layout()
    _save(fig, "tenure_distribution.png", save_dir)


def plot_monthly_charges(df: pd.DataFrame, save_dir: str = "images") -> None:
    """Boxplot + violin for MonthlyCharges vs churn."""
    df = df.copy()
    df["Churn_label"] = df["Churn"].map({0: "Stayed", 1: "Churned"})
    palette = {"Stayed": TEAL, "Churned": ORANGE}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=df, x="Churn_label", y="MonthlyCharges", palette=palette,
                order=["Stayed", "Churned"], ax=axes[0])
    axes[0].set_title("Monthly charges — boxplot", fontweight="bold")
    axes[0].set_xlabel("")
    sns.despine(ax=axes[0])

    sns.violinplot(data=df, x="Churn_label", y="MonthlyCharges", palette=palette,
                   order=["Stayed", "Churned"], inner="quartile", ax=axes[1])
    axes[1].set_title("Monthly charges — violin", fontweight="bold")
    axes[1].set_xlabel("")
    sns.despine(ax=axes[1])

    fig.tight_layout()
    _save(fig, "monthly_charges_vs_churn.png", save_dir)


def plot_payment_vs_churn(df: pd.DataFrame, save_dir: str = "images") -> None:
    """Horizontal bar: churn rate per payment method."""
    pay = df.groupby("PaymentMethod")["Churn"].mean().sort_values()
    colors = [TEAL if v < 0.30 else AMBER if v < 0.40 else ORANGE for v in pay.values]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(pay.index, pay.values * 100, color=colors, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}%", va="center", fontweight="bold")
    ax.set_title("Churn rate by payment method", fontweight="bold", pad=12)
    ax.set_xlabel("Churn rate (%)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlim(0, 55)
    sns.despine(ax=ax)
    fig.tight_layout()
    _save(fig, "payment_vs_churn.png", save_dir)


def plot_all(df: pd.DataFrame, save_dir: str = "images") -> None:
    """Run all EDA plots in one call."""
    plot_churn_rate(df, save_dir)
    plot_contract_vs_churn(df, save_dir)
    plot_tenure_distribution(df, save_dir)
    plot_monthly_charges(df, save_dir)
    plot_payment_vs_churn(df, save_dir)
    print("All EDA plots saved.")


if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_churn.csv")
    plot_all(df)