# Design Decisions

This document records the reasoning behind Stockast's non-obvious architectural choices — the "why," not just the "what." Each of these was a deliberate decision made (and in several cases, corrected) during development, not an accident of how the code turned out.

---

## 1. The canonical schema

Every user's dataset has different column names, orders, and formats. Rather than writing dataset-specific logic anywhere downstream, Stockast defines one fixed internal schema (`src/common/canonical_schema.py`) — 9 fields, split into `REQUIRED` (`date`, `product_id`, `quantity_sold`) and `OPTIONAL` (`product_name`, `category`, `unit_price`, `unit_cost`, `warehouse_id`, `current_stock`, `lead_time_days`).

**Every module past `schema_mapping` only ever talks to this schema.** This is what makes the platform genuinely generic: adding a new analysis module never requires touching ingestion, mapping, or validation — it just consumes the same 9 fields everyone else does.

## 2. Hybrid schema mapping — auto-suggest, human confirms

Fully automatic column mapping is fast but risky: one wrong guess (e.g. matching `returns` to `quantity_sold`) silently corrupts every downstream KPI. Fully manual mapping is safe but tedious.

Stockast uses fuzzy string matching (`rapidfuzz`) against known aliases per field to **suggest** a mapping, with genuine ambiguity (two equally-strong candidate columns) explicitly flagged — but a human always confirms before anything is applied to real data. `MappingResult.is_confirmed` is checked and enforced in code (`MappingNotConfirmedError`), not just by UI convention.

## 3. Optional fields are never fabricated

If a dataset has no `unit_cost`, no code anywhere invents a plausible-looking value. Cost-dependent KPIs (margin-based ABC analysis, profit calculations) are explicitly marked unavailable instead. The one deliberate exception is `warehouse_id`, which defaults to `"main"` when absent — because a single synthetic warehouse name is a safe structural default (every downstream `groupby("warehouse_id")` still works correctly), whereas a fabricated cost or price would be a false business fact.

## 4. Outliers are flagged, never dropped

An unusually large `quantity_sold` value could be a data-entry error — or it could be a real demand spike (a product going viral, a seasonal event). Validation uses the IQR method to flag statistical outliers as **warnings**, and `data_cleaning` deliberately never removes them. The distinction between "typo" and "real event" requires business context a statistical rule can't supply — so the platform surfaces it for a human to judge rather than deciding unilaterally.

## 5. The `field_is_available` bug, and why it mattered

Early in Phase 13, every optional-field check across `eda`, `analytics`, `kpis`, and `price_elasticity` used `field_is_available(df.columns, field)` — checking only whether a *column* existed. Because `apply_mapping()` always creates all 9 canonical columns (filling unmapped ones with `None`), this check **always returned `True`** post-mapping, even for entirely empty fields. A dataset with no `unit_cost` data would still attempt cost-based calculations on an all-null column.

The fix: `field_is_available` now checks for at least one real, non-null value (`df[field].notna().any()`), not just column presence. This is a good example of a bug that only surfaces at integration time — every individual module's unit tests passed, because their fixtures never modeled the "column exists but is entirely null" case that only `apply_mapping()`'s real behavior produces.

## 6. Seasonality: minimum cycle count, verified empirically

The original seasonality check required 2 full cycles of data before testing a period (e.g. 14 days for weekly). Testing this against pure random noise revealed a real problem: at 2–4 cycles, the false-positive rate for "detecting" a fake weekly pattern in random data ranged from 47% to 100% across 30 test seeds. The requirement was raised to 6 cycles (42 days for weekly, 180 for monthly), which brought the false-positive rate down to roughly 7%. This number came from actually testing the claim, not from a rule of thumb.

## 7. ML risk prediction: real labels, or an honest fallback — never a fake middle ground

Predicting stockout risk needs real historical examples of stockouts to learn from. Since no dataset provides an explicit "stockout occurred" label, one is constructed by scanning each product's `current_stock` history (forward-filled to daily granularity) for genuine `stock > 0 → stock ≤ 0` transitions.

Two things had to be true before attempting a real trained model:
- Enough products have genuine historical stock variation (not a single static snapshot)
- Enough real stockout events exist dataset-wide (≥30) to train on

If either check fails, `inventory_ml` reports the model as **unavailable** with a specific reason — it does not silently substitute a simpler formula and present it as the same thing. A separate, explicitly labeled rule-based risk score (combining days-of-inventory, demand volatility, and forecast trend) lives in `recommendation_engine` as a distinct, honestly-presented fallback, not folded into `inventory_ml`'s output.

## 8. Walk-forward, point-in-time ML features — not a single static label per product

An earlier version of the risk model computed one feature row per product, labeled by whether that product *ever* stocked out across its entire history — a fundamentally different (and much weaker) question than "will this product stock out in the next 14 days, given what we know right now." 

The corrected design builds multiple snapshots per product through its history (one every 7 days), computing features only from data *before* that snapshot and labeling only from what happens in the following 14 days — genuinely testing the intended forward-looking question, with no leakage from the future into the features.

## 9. Forecasting: comparing techniques honestly, with a real baseline

Naive (seasonal or moving-average, chosen based on whether real seasonality was detected — never assumed), Exponential Smoothing, and ARIMA are all fit on a training split and evaluated against held-back real data (MAE), per product. The winning technique is then retrained on the full history for the actual forecast. This means the "best" model is never chosen by inspection or intuition — it's whichever one was verifiably most accurate on data it hadn't seen.

## 10. The AI layer never reasons independently

`ai_assistant` is deliberately the simplest module in the platform: it only ever receives already-computed, already-labeled facts (a risk label our own code decided, a recommendation our own rules generated) and turns them into readable prose. The prompt explicitly instructs the model not to second-guess the numbers or labels it's given. This was a constraint set at the very start of the project, specifically to keep the platform's conclusions consistent and reproducible — an LLM asked to *interpret* raw numbers could reasonably disagree with itself between calls; one asked only to *phrase* an already-fixed conclusion cannot.

## 11. Everything degrades gracefully, by construction

A recurring pattern across every module: check whether the data actually supports a computation before attempting it, and return a clear, structured "unavailable" result rather than crashing or guessing when it doesn't. This is why the platform works the same way — never breaking, always explaining itself — whether the input is a rich, 100-field enterprise export or a bare 3-column CSV with only `date`, `product_id`, and `quantity_sold`.