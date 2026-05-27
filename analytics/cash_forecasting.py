import os
import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Use non-interactive Agg backend for matplotlib to prevent Tkinter window issues in background tasks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_historical_cash_data(
    months: int = 18,
    initial_cash: float = 5000000.0,
    monthly_burn: float = 220000.0,
    noise_std: float = 40000.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic Pandas DataFrame representing historical cash balances
    over a set number of months, assuming a steady operational burn rate.
    """
    np.random.seed(seed)
    
    # Generate monthly dates starting from (months - 1) months ago up to today
    today = datetime.date.today()
    date_list = []
    for i in range(months):
        # Approximate monthly increments by subtracting 30 days
        dt = today - datetime.timedelta(days=30 * (months - 1 - i))
        date_list.append(dt)
        
    # Generate cash balances with a steady burn rate and noise
    cash_balances = []
    current_cash = initial_cash
    
    for i in range(months):
        noise = np.random.normal(0, noise_std)
        balance = current_cash + noise
        # Cash cannot go below zero
        balance = max(0.0, balance)
        cash_balances.append(balance)
        # Burn cash for the next month
        current_cash -= monthly_burn

    df = pd.DataFrame({
        "Date": pd.to_datetime(date_list),
        "Cash_Balance": cash_balances
    })
    return df

def forecast_cash_runway(df: pd.DataFrame, forecast_days: int = 365) -> tuple[pd.DataFrame, datetime.date, float]:
    """
    Fits a linear trend regression on historical cash balances,
    estimates the exact exhaustion date (Zero-Balance Point),
    and returns forecasting points.
    """
    # 1. Prepare numerical feature representing elapsed days from the start
    start_date = df["Date"].min()
    df["Days"] = (df["Date"] - start_date).dt.days
    
    X_train = df[["Days"]].values
    y_train = df["Cash_Balance"].values
    
    # 2. Fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # 3. Calculate Zero-Balance Date ( runway intersection with $0 )
    # y = slope * x + intercept  =>  0 = slope * x + intercept  =>  x = -intercept / slope
    if slope >= 0:
        # Cash is increasing or flat
        exhaustion_date = None
        runway_months = float('inf')
    else:
        zero_days = -intercept / slope
        exhaustion_date = (start_date + datetime.timedelta(days=int(zero_days))).date()
        
        # Calculate runway remaining from today
        today = datetime.date.today()
        days_remaining = (exhaustion_date - today).days
        runway_months = max(0.0, days_remaining / 30.4)
    
    # 4. Generate forecast timeline DataFrame
    max_days = int(zero_days) if (slope < 0 and zero_days > df["Days"].max()) else df["Days"].max() + forecast_days
    forecast_days_array = np.arange(0, max_days + 1)
    forecast_dates = [start_date + datetime.timedelta(days=int(d)) for d in forecast_days_array]
    
    predicted_cash = model.predict(forecast_days_array.reshape(-1, 1))
    # Clip predictions at 0 for logical consistency
    predicted_cash = np.clip(predicted_cash, 0, None)
    
    forecast_df = pd.DataFrame({
        "Date": pd.to_datetime(forecast_dates),
        "Forecast_Balance": predicted_cash
    })
    
    return forecast_df, exhaustion_date, runway_months

def generate_forecast_chart(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    exhaustion_date: datetime.date,
    output_path: str = "analytics/cash_runway_forecast.png"
):
    """
    Generates and saves a publication-quality visualization showing historical balance,
    forecasted trend line, and the predicted point of runway exhaustion.
    """
    # Create target directories if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot historical data
    plt.plot(
        historical_df["Date"],
        historical_df["Cash_Balance"] / 1000, # Show in Thousands ($k)
        label="Historical Cash Balance",
        color="#1a73e8",
        linewidth=2.5,
        marker="o",
        markersize=5
    )
    
    # Plot forecasted data
    plt.plot(
        forecast_df["Date"],
        forecast_df["Forecast_Balance"] / 1000,
        label="Forecasted Trend (Linear Regression)",
        color="#ea4335",
        linestyle="--",
        linewidth=2
    )
    
    # Mark the exhaustion point if it exists
    if exhaustion_date:
        exhaustion_dt = pd.to_datetime(exhaustion_date)
        plt.scatter(
            [exhaustion_dt],
            [0],
            color="#d93025",
            s=120,
            zorder=5,
            marker="X",
            label=f"Exhaustion Date: {exhaustion_date.strftime('%Y-%m-%d')}"
        )
        # Vertical reference line
        plt.axvline(x=exhaustion_dt, color="#ea4335", linestyle=":", alpha=0.6)
        
    plt.title("Cash Runway Forecasting & Runway Depletion Projection", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Timeline Date", fontsize=12)
    plt.ylabel("Cash Balance ($ in Thousands)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, shadow=True)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Visualization plot saved successfully at: {output_path}")

if __name__ == "__main__":
    print("====================================================")
    print("Running Standalone Cash Forecasting Engine...")
    print("====================================================")

    # 1. Generate historical cash data (18 months)
    history_df = generate_historical_cash_data(months=18)
    print(f"Ingested {len(history_df)} months of historical cash balance records.")
    print("Historical Balance Samples:")
    print(history_df.tail(3).to_string(index=False))
    
    # 2. Run forecasting
    forecast_df, depletion_date, remaining_runway = forecast_cash_runway(history_df)
    
    print("\n--- Quantitative Analysis Summary ---")
    if depletion_date:
        print(f"Predicted Point of Zero Cash: {depletion_date.strftime('%B %d, %Y')}")
        print(f"Remaining Cash Runway:        {remaining_runway:.2f} months")
    else:
        print("Runway status is stable / cash reserves are growing.")
    print("--------------------------------------")
    
    # 3. Save plot visual
    chart_file = "analytics/cash_runway_forecast.png"
    generate_forecast_chart(history_df, forecast_df, depletion_date, chart_file)
