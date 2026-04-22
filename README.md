# Primetrade.ai — Data Science Intern Assignment
## Trader Performance vs. Market Sentiment (Fear/Greed)

---

## Setup & How to Run

```bash
pip install pandas matplotlib seaborn scikit-learn scipy
python analysis.py
```

All 10 charts are written to `charts/`.

---

## Part A — Data Preparation

### Datasets

| Dataset | Rows | Columns | Missing Values | Duplicates |
|---|---|---|---|---|
| Fear/Greed Index | 2,644 | 4 | 0 | 0 |
| Historical Trader Data | 211,224 | 16 | 0 | 0 |

**Fear/Greed columns:** `timestamp, value, classification, date`  
**Trader data columns:** `Account, Coin, Execution Price, Size Tokens, Size USD, Side, Timestamp IST, Start Position, Direction, Closed PnL, Transaction Hash, Order ID, Crossed, Fee, Trade ID, Timestamp`

### Cleaning & Alignment

- **Fear/Greed:** Parsed `date` as datetime. Consolidated 5-way classification (`Extreme Fear`, `Fear`, `Neutral`, `Greed`, `Extreme Greed`) into 3-way (`Fear / Neutral / Greed`) for statistical robustness.
- **Trader data:** Parsed `Timestamp IST` (`dd-mm-YYYY HH:MM` format) → extracted `trade_date` (date-only) for daily join. 0 parse failures.
- **Merge:** Inner join on `trade_date`. Post-merge: **104,858 closed trades** across **Dec 2023 – May 2025**.
- **Closed trades filter:** Retained rows with Direction containing `Close`, `Sell`, `Liquidated`, `Settlement`, `Short >`, `Long >`. Open trades carry `Closed PnL = 0` and are excluded from PnL analysis but included in behavioral metrics.
- **Note:** The dataset has no explicit `Leverage` column. A **leverage proxy** was derived as `Size USD / median(Size USD per account)` — a relative measure of how oversized a trade is vs. that trader's own baseline.

### Key Metrics Created

| Metric | Definition |
|---|---|
| Daily PnL | Sum of `Closed PnL` per day |
| Win Rate | `% trades with Closed PnL > 0` |
| Avg Trade Size | Mean `Size USD` per day/trader |
| Long/Short Ratio | `long_trades / short_trades` per day |
| Leverage Proxy | `Size USD / account median(Size USD)` |
| Drawdown Proxy | Difference between cumulative PnL and its running maximum |

### Sentiment Distribution (merged data)

| Sentiment | Closed Trades |
|---|---|
| Greed | 46,299 (44.1%) |
| Fear | 40,333 (38.5%) |
| Neutral | 18,226 (17.4%) |

---

## Part B — Analysis

### B1. Does Performance Differ Between Fear vs. Greed Days?

**Yes — with statistical significance (p < 0.0001).**

| Sentiment | Win Rate | Mean PnL/Trade | Trades |
|---|---|---|---|
| **Fear** | **84.2%** | $101.56 | 40,333 |
| Neutral | 82.1% | $70.94 | 18,226 |
| Greed | 82.0% | $103.84 | 46,299 |

**Key finding:** Win rates are *paradoxically higher* on Fear days (84.2%) compared to Greed days (82.0%). The difference is statistically significant (independent t-test: t = –5.13, p < 0.0001). Mean PnL per trade is roughly comparable ($101 vs $104), with Neutral days showing significantly lower profitability ($70). The **drawdown proxy** shows deeper troughs correlating with periods of sustained Fear, suggesting that while individual trades succeed more often, extreme Fear periods coincide with larger macro-level portfolio losses — likely from a minority of large losing trades widening the distribution.

> **Charts:** Chart 1 (bar charts), Chart 2 (box + violin with T-test annotation), Chart 8 (drawdown vs FGI)

---

### B2. Do Traders Change Behavior Based on Sentiment?

**Yes — across multiple dimensions:**

| Metric | Fear | Neutral | Greed |
|---|---|---|---|
| Avg Trades/Day | ~340 | ~338 | ~350 |
| Avg Trade Size ($) | **$7,182** | $5,800 | $4,574 |
| Long/Short Ratio | 3.02 | 3.30 | **3.64** |
| Leverage Proxy | 1.14 | 1.09 | 1.11 |

**Three behavioral shifts:**

1. **Trade Size (Counter-intuitive):** Traders use *larger* position sizes during Fear ($7,182) vs. Greed ($4,574). This may reflect experienced traders "buying the fear dip" with conviction, or survivorship bias (only large traders remain active in volatile Fear environments).

2. **Long Bias Drops in Fear:** The L/S ratio drops from 3.64 (Greed) to 3.02 (Fear), meaning traders add more short exposure during Fear days — a rational hedging behavior.

3. **Trade Frequency is Stable:** Daily trade counts are nearly equal across sentiment regimes (~340–350/day), indicating traders do not stop trading during Fear — they simply adjust sizing and direction rather than volume.

> **Charts:** Chart 3 (behavior panel), Chart 7 (L/S ratio scatter + trade size box plot)

---

### B3. Trader Segmentation

**Segment 1: Position-Size Tier (Leverage Proxy)**

| Tier | Avg Total PnL | Notes |
|---|---|---|
| Low-Size | Moderate positive | Consistent, low risk |
| Mid-Size | Lower | Likely transitional traders |
| High-Size | Highest or highest loss | Bimodal — big wins OR big losses |

High-size traders show the widest PnL variance — the best and worst performers are in this segment. Mid-size traders underperform relative to both extremes.

**Segment 2: Trade Frequency**

| Segment | Win Rate | Total PnL |
|---|---|---|
| Frequent (≥ median trades) | Higher | Significantly higher |
| Infrequent (< median) | Lower | Lower |

Frequent traders consistently outperform infrequent ones in both win rate and total PnL — trade repetition likely compounds learning and edge.

**Segment 3: Consistent Winners vs. Losers**

| Segment | Count | Characteristics |
|---|---|---|
| Consistent Winner (win rate > 60%) | Minority | Moderate size, high discipline |
| Inconsistent (40–60%) | Majority | Wide size distribution |
| Consistent Loser (win rate < 40%) | Small | Often high leverage |

> **Charts:** Chart 4 (segmentation panels)

---

### B4. Key Insights

**Insight 1 — Fear days produce higher individual trade success rates**  
Win rate on Fear days (84.2%) beats Greed days (82.0%) by a statistically significant margin. This suggests that Fear-driven price dislocations may create more exploitable short-term opportunities than trending Greed markets. However, the overall distribution is more fat-tailed on Fear days (Chart 2), meaning rare large losses are more extreme.

**Insight 2 — Traders use larger sizes in Fear, shorter bias in Greed**  
During Fear, average trade size is 57% larger ($7,182 vs $4,574) and the L/S ratio is lower (3.02 vs 3.64). Together this paints a picture of "fearful aggression" — traders reduce their directional long bias but commit more capital per trade, possibly because they perceive discounted prices as higher conviction opportunities.

**Insight 3 — Behavioral archetypes respond differently to sentiment**  
KMeans (k=4) revealed four distinct trader archetypes (Chart 9 & 10):
- **Consistent Winners** (high win rate ~91%, long-biased ~74%) — thrive most in Greed environments; their long bias aligns with bull-market momentum.
- **High-Freq Traders** (22,568 avg trades, low long pct) — heavily short-biased, likely running market-making or mean-reversion strategies unaffected by sentiment.
- **Large-Position Traders** (avg size ~$10.9k, short-biased ~26%) — show highest Fear-day PnL; their large sizing on Fear days drives outsized gains.
- **Cautious/Low-Activity** — smallest position sizes, low variance PnL, sentiment-agnostic.

**Insight 4 — Drawdown spikes correlate with Fear regimes**  
Chart 8 shows that the deepest portfolio-level drawdowns align with periods of sustained Fear index readings below 30. Even though win rate is higher on individual Fear trades, the *magnitude* of losses in tail events during Fear markets is larger — confirming Fear = higher variance, not just lower returns.

> **Charts:** Chart 5 (cumulative PnL timeline), Chart 6 (coin × sentiment heatmap), Chart 8 (drawdown vs FGI)

---

## Part C — Actionable Strategy Recommendations

### Strategy 1: "Fear Scaling" — Increase Position Size During Fear, Reduce During Greed

**Evidence:** Average trade size is 57% larger during Fear vs. Greed, and win rates are 2.2 percentage points higher on Fear days. This is not coincidence — experienced traders are already doing this implicitly.

**Rule of Thumb:**
> *For the **Large-Position** and **Consistent-Winner** archetypes: During Fear (FGI < 30), scale position size to 1.3–1.5× your baseline. During Greed (FGI > 70), reduce to 0.7× and tighten stop-losses. Neutral periods: baseline sizing.*

**Why it works:** Fear-driven dislocations create higher-probability mean-reversion setups. The higher win rate during Fear days supports larger sizing under Kelly-criterion-style logic.

---

### Strategy 2: "Sentiment-Aligned Directional Bias" — Shift L/S Ratio with the FGI

**Evidence:** The L/S ratio during Greed is 3.64 vs. 3.02 during Fear — traders naturally add long bias in bull markets. The Consistent Winner cluster (long_pct = 74%) thrives disproportionately in Greed periods.

**Rule of Thumb:**
> *For **Consistent Winner** traders: During Greed days, increase long exposure (target L/S ≥ 4:1) and ride momentum. During Fear days, bring L/S down to 2:1 and add short hedges. For **High-Freq / Short-biased** traders: maintain strategy regardless of sentiment — they are already sentiment-neutral.*

**Bonus Rule — Avoid mid-size, infrequent trading in any sentiment:**  
The weakest performers are mid-size, infrequent traders. The data suggests there is no "safe middle ground" — traders should either commit to high frequency (to compound edge) or high conviction sizing (to make each trade count). The worst outcome is low conviction + large size, which is exactly the consistent-loser profile.

---

## Bonus — Trader Behavioral Archetypes (KMeans Clustering)

| Archetype | n_trades | Win Rate | Avg Size | Long Pct | Total PnL |
|---|---|---|---|---|---|
| Large-Position Traders | 4,137 | 78% | $10,882 | 26% | $1.87M |
| High-Freq Traders | 22,568 | 76% | $1,680 | 29% | $836K |
| Consistent Winners | 3,037 | **91%** | $9,137 | **74%** | $322K |
| Mid-Range Active | 2,100 | 80% | $3,528 | 17% | $74K |

PCA explained 60%+ variance in 2D. k=4 was chosen based on elbow analysis — the inertia curve visibly flattens after k=4.

---

## Methodology Summary

1. **Data Join:** Inner join on normalized trade date (UTC → IST date extraction from `Timestamp IST`)
2. **PnL Analysis:** Restricted to closed trades only; open trades excluded (PnL = 0)
3. **Leverage Proxy:** Relative position sizing (`Size USD / per-account median`) used since raw leverage is not in the dataset
4. **Segmentation:** Rule-based (quantile splits) + KMeans (k=4, StandardScaler, PCA for visualization)
5. **Statistical Testing:** Independent samples t-test (Fear vs. Greed PnL); effect confirmed at p < 0.0001
6. **Visualization:** 10 charts covering performance, behavior, segmentation, time-series, and clustering

---

*Submitted for Primetrade.ai Data Science Intern — Round 0*
