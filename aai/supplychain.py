import pandas as pd
import numpy as np
from collections import deque, defaultdict

# ----------------------------
# Config (tweak freely)
# ----------------------------
ORDER_COST_FIXED = 5.0  # fixed cost per PO (set 0 if not wanted)
ALPHA = 0.3             # EWMA smoothing factor for demand
SEED = 42

np.random.seed(SEED)

# ----------------------------
# Data loading helpers
# ----------------------------
def load_data(sales_path="sales.csv", inv_path="inventory.csv", params_path="params.csv"):
    sales = pd.read_csv(sales_path, parse_dates=["date"])
    inv = pd.read_csv(inv_path)
    params = pd.read_csv(params_path)
    return sales, inv, params

def ensure_complete_calendar(sales):
    # Fill missing dates per SKU with qty_sold=0
    skus = sales["sku"].unique()
    date_index = pd.date_range(sales["date"].min(), sales["date"].max(), freq="D")
    frames = []
    for sku in skus:
        s = sales.loc[sales["sku"] == sku, ["date", "qty_sold"]].set_index("date").reindex(date_index, fill_value=0)
        s["sku"] = sku
        s = s.rename_axis("date").reset_index()
        s = s.rename(columns={"index": "date"})
        frames.append(s)
    return pd.concat(frames, ignore_index=True)

# ----------------------------
# Forecasting (EWMA) + error SD
# ----------------------------
def ewma_forecast(series, alpha=ALPHA):
    """Return point forecasts and residual std (rolling)."""
    f = []
    prev = series.iloc[0]  # seed with first observation
    f.append(prev)
    for x in series.iloc[1:]:
        prev = alpha * x + (1 - alpha) * prev
        f.append(prev)
    fc = pd.Series(f, index=series.index)
    residuals = series - fc
    # Use expanding std (avoid 0)
    res_sd = residuals.expanding().std().fillna(residuals.std() if not np.isnan(residuals.std()) else 0.0)
    res_sd = res_sd.replace([np.nan, np.inf, -np.inf], 0.0)
    return fc, res_sd

def z_from_service_level(p):
    # Approx normal inverse (rough): use scipy if allowed; here a small lookup
    # Common levels
    table = {0.80: 0.84, 0.85: 1.04, 0.90: 1.28, 0.95: 1.65, 0.97: 1.88, 0.98: 2.05, 0.99: 2.33}
    # Fallback linear-ish
    keys = sorted(table.keys())
    if p in table: return table[p]
    if p <= keys[0]: return table[keys[0]]
    if p >= keys[-1]: return table[keys[-1]]
    # interpolate
    for i in range(len(keys)-1):
        if keys[i] < p < keys[i+1]:
            w = (p-keys[i])/(keys[i+1]-keys[i])
            return table[keys[i]]*(1-w) + table[keys[i+1]]*w

# ----------------------------
# Agent loop
# ----------------------------
def simulate_agent(sales, inv, params):
    # Merge params for quick lookup
    pmap = {r["sku"]: r for _, r in params.iterrows()}
    start_stock = {r["sku"]: r["opening_stock"] for _, r in inv.iterrows()}

    # Prepare per-SKU frames
    results = []
    logs = []
    for sku in sales["sku"].unique():
        s = sales[sales["sku"] == sku].sort_values("date").reset_index(drop=True)
        qty = s["qty_sold"].astype(float)

        # Forecast demand (EWMA)
        fc, res_sd = ewma_forecast(qty)

        # Params
        unit_cost = float(pmap[sku]["unit_cost"])
        hold_cost = float(pmap[sku]["holding_cost_per_day"])
        stockout_cost = float(pmap[sku]["stockout_cost"])
        lead = int(pmap[sku]["lead_time_days"])
        moq = int(pmap[sku]["min_order_qty"])
        service = float(pmap[sku]["service_level"])
        z = z_from_service_level(service)

        # Initialize state
        on_hand = float(start_stock.get(sku, 0))
        pipeline = deque()  # (arrival_day_idx, qty)
        day_costs = []
        fulfilled = 0
        demanded = 0

        # Simple safety stock approximation:
        # safety = z * sigma_demand * sqrt(lead_time)
        # Use expanding residual sd as sigma proxy
        safety_by_day = z * res_sd * np.sqrt(max(1, lead))

        for i, row in s.iterrows():
            date = row["date"]
            demand = float(row["qty_sold"])
            forecast_today = float(fc.iloc[i])

            # Receive any due orders
            while pipeline and pipeline[0][0] == i:
                arrived = pipeline.popleft()[1]
                on_hand += arrived
                logs.append(f"{date} [{sku}] → Received {arrived} units (pipeline). On-hand={on_hand:.1f}")

            # Serve demand
            shipped = min(on_hand, demand)
            on_hand -= shipped
            lost_sales = max(0.0, demand - shipped)

            # Costs
            holding = on_hand * hold_cost
            stockout = lost_sales * stockout_cost
            day_cost = holding + stockout
            day_costs.append(day_cost)

            fulfilled += shipped
            demanded += demand

            # Decide reorder based on reorder point:
            # ROP = lead_time * forecast + safety
            # Use today's forecast as proxy for daily rate
            safety = float(safety_by_day.iloc[i]) if not np.isnan(safety_by_day.iloc[i]) else 0.0
            reorder_point = lead * max(0.0, forecast_today) + safety

            # Projected position (on_hand + pipeline arriving before stockout is tricky; we’ll use on_hand only for simplicity)
            projected_position = on_hand

            if projected_position < reorder_point:
                # Order up-to (S): target = lead*forecast + safety + review buffer (1 day of forecast)
                target_level = reorder_point + max(1.0, forecast_today)
                order_qty = max(moq, int(np.ceil(target_level - projected_position)))
                arrival_day = i + lead
                pipeline.append((arrival_day, order_qty))
                day_cost += ORDER_COST_FIXED
                day_costs[-1] = day_cost  # include order cost today
                logs.append(
                    f"{date} [{sku}] ROP={reorder_point:.1f}, on_hand={on_hand:.1f} ⇒ ORDER {order_qty} (arrives day+{lead}). "
                    f"Reason: forecast={forecast_today:.2f}, safety={safety:.2f}"
                )

        total_cost = sum(day_costs)
        fill_rate = fulfilled / max(1.0, demanded)
        results.append({
            "sku": sku,
            "period_days": len(s),
            "total_demand": demanded,
            "fulfilled": fulfilled,
            "fill_rate": round(fill_rate, 4),
            "ending_on_hand": round(on_hand, 2),
            "total_cost": round(total_cost, 2),
            "avg_daily_cost": round(total_cost / len(s), 2)
        })

    return pd.DataFrame(results), logs

# ----------------------------
# Demo data generator (optional)
# ----------------------------
def generate_demo_data():
    dates = pd.date_range("2025-01-01", periods=90, freq="D")
    skus = ["A100", "B200"]
    sales_rows = []
    for sku in skus:
        base = 20 if sku == "A100" else 12
        season = np.sin(np.linspace(0, 3*np.pi, len(dates))) * (4 if sku=="A100" else 2)
        noise = np.random.normal(0, 3, len(dates))
        demand = np.maximum(0, np.round(base + season + noise)).astype(int)
        for d, q in zip(dates, demand):
            sales_rows.append({"date": d, "sku": sku, "qty_sold": int(q)})
    pd.DataFrame(sales_rows).to_csv("sales.csv", index=False)
    pd.DataFrame([
        {"sku": "A100", "opening_stock": 250},
        {"sku": "B200", "opening_stock": 180},
    ]).to_csv("inventory.csv", index=False)
    pd.DataFrame([
        {"sku":"A100","unit_cost":10,"holding_cost_per_day":0.02,"stockout_cost":2.0,"lead_time_days":5,"min_order_qty":50,"service_level":0.95},
        {"sku":"B200","unit_cost":8,"holding_cost_per_day":0.015,"stockout_cost":1.5,"lead_time_days":7,"min_order_qty":40,"service_level":0.95},
    ]).to_csv("params.csv", index=False)

if __name__ == "__main__":
    # Generate demo data once if you don't have CSVs
    generate_demo_data()

    sales, inv, params = load_data()
    sales = ensure_complete_calendar(sales)
    summary, log = simulate_agent(sales, inv, params)

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== SAMPLE LOG (first 30 lines) ===")
    for line in log[:30]:
        print(line)
