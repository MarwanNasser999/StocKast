"""
Prompt templates for src.ai_assistant.

Every template only ever includes ALREADY-COMPUTED structured facts --
never raw data. The LLM's only job is to phrase these facts as
professional prose; every judgment (risk label, recommended action,
elasticity classification) was already decided by our own code.
"""

from __future__ import annotations

PRODUCT_REPORT_PROMPT = """You are writing a brief, professional inventory status update for a business manager.
Use ONLY the facts provided below. Do not invent numbers, do not make assumptions beyond what is given,
and do not second-guess the risk level or recommendation provided -- these were already determined by
statistical and machine learning analysis.

Product: {display_name}
Days of inventory remaining: {days_of_inventory}
Risk level: {risk_label} (risk score: {risk_score})
Demand trend: {trend_description}
Recommended action: {action}
Reasoning for recommendation: {reasoning}

Write 2-3 sentences summarizing this product's situation and the recommended action, in a tone
appropriate for a business manager reading a daily operations report."""


EXECUTIVE_SUMMARY_PROMPT = """You are writing a brief executive summary of inventory health for a business owner.
Use ONLY the facts provided below. Do not invent numbers or make claims beyond what is given.

Total products analyzed: {total_products}
Products at high risk of stockout: {high_risk_count}
Products flagged as dead or slow-moving stock: {dead_stock_count}
Total recommended actions: {total_recommendations}

Top 3 highest-priority recommendations:
{top_recommendations}

Write a 3-4 sentence executive summary covering the overall inventory health and the most
urgent items needing attention, in a tone appropriate for a business owner reviewing a
weekly report."""