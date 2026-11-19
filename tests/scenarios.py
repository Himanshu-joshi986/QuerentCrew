"""Reusable scripted LLM conversations for node and graph tests."""

from querentcrew.schemas import AnalysisPlan, ChartSpec, Insights, RouteDecision, SQLDraft

Q2_Q3_BY_REGION = """
SELECT r.name AS region,
       ROUND(SUM(CASE WHEN o.order_date BETWEEN '2025-04-01' AND '2025-06-30'
                 THEN oi.quantity*oi.unit_price*(1-oi.discount) END), 2) AS q2_revenue,
       ROUND(SUM(CASE WHEN o.order_date BETWEEN '2025-07-01' AND '2025-09-30'
                 THEN oi.quantity*oi.unit_price*(1-oi.discount) END), 2) AS q3_revenue
FROM orders o JOIN order_items oi ON oi.order_id = o.order_id
JOIN customers c ON c.customer_id = o.customer_id JOIN regions r ON r.region_id = c.region_id
WHERE o.status = 'completed' GROUP BY r.name
"""

SOUTH_BY_CATEGORY = """
SELECT p.category,
       ROUND(SUM(CASE WHEN o.order_date BETWEEN '2025-04-01' AND '2025-06-30'
                 THEN oi.quantity*oi.unit_price*(1-oi.discount) END), 2) AS q2_revenue,
       ROUND(SUM(CASE WHEN o.order_date BETWEEN '2025-07-01' AND '2025-09-30'
                 THEN oi.quantity*oi.unit_price*(1-oi.discount) END), 2) AS q3_revenue
FROM orders o JOIN order_items oi ON oi.order_id = o.order_id
JOIN products p ON p.product_id = oi.product_id
JOIN customers c ON c.customer_id = o.customer_id JOIN regions r ON r.region_id = c.region_id
WHERE o.status = 'completed' AND r.name = 'South' GROUP BY p.category
"""


def why_drop_script():
    """2-step plan; the first SQL has a typo so the self-correction loop is exercised."""
    return {
        "router": RouteDecision(intent="data_question", reason="needs revenue numbers"),
        "planner": AnalysisPlan(reasoning="compare quarters then drill into categories",
                                sub_questions=["Q2 vs Q3 2025 revenue by region",
                                               "Q2 vs Q3 2025 revenue by category in the South region"]),
        "sql_writer": [
            SQLDraft(reasoning="typo", sql="SELECT regoin FROM regions"),
            lambda system, user, schema: SQLDraft(
                reasoning="fixed after error" if "no such column" in user else "unexpected",
                sql=Q2_Q3_BY_REGION),
            SQLDraft(reasoning="category drilldown", sql=SOUTH_BY_CATEGORY),
        ],
        "analyst": Insights(key_findings=["South revenue fell sharply from Q2 to Q3"], caveats=[]),
        "charter": ChartSpec(chart_type="bar", result_index=0, x="region", y="q3_revenue", title="Q3 revenue"),
        "reporter": "**South revenue dropped** in Q3, mainly Electronics.",
    }


def pii_script():
    return {
        "router": RouteDecision(intent="data_question", reason="customer lookup"),
        "planner": AnalysisPlan(reasoning="one query", sub_questions=["emails of 3 Enterprise customers"]),
        "sql_writer": SQLDraft(reasoning="r",
                               sql="SELECT name, email FROM customers WHERE segment = 'Enterprise' LIMIT 3"),
        "analyst": Insights(key_findings=["3 customers listed"], caveats=[]),
        "charter": ChartSpec(chart_type="none"),
        "reporter": "Here are the customers.",
    }
