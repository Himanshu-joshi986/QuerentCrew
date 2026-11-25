# Re-grading of the saved answers

Model: `groq:openai/gpt-oss-120b`, temperature 0.0, 360 saved answers (`evals/results/checkpoint.jsonl`). No model was called.

Command: `python evals/regrade.py`

Input files on disk match the fingerprints stored in the checkpoint (line endings ignored).

## 1. Reproducibility check

Re-running the **original grader** on the saved SQL gives the recorded verdict for **360 of 360** answers.

## 2. Grader validation (mutation testing)

Each gold query is changed automatically. *Broken queries* change the meaning and must be rejected; *harmless rewrites* keep the meaning and must be accepted (`src/querentcrew/grader_validation.py`).

| grader | gold vs itself | broken queries rejected | harmless rewrites accepted | broken queries with an identical result (no grader can tell) |
|---|---|---|---|---|
| original (round1) | 40/40 | 185/189 | 121/126 | 9 |
| corrected (tolerance) | 40/40 | 189/189 | 126/126 | 9 |

## 3. Final accuracy

Mean over 3 runs. *95% interval* is a Wilson interval over the 20 questions of the suite (each question scored as its mean over the runs), so repeated runs are not treated as extra questions. *Run range* is the lowest and highest single run.

### Suite: dev

| prompt | as recorded in the run | original grader, re-run | corrected grader | corrected grader + 5 extra databases | manual review, ambiguous = wrong | manual review, ambiguous = correct | 95% interval (corrected grader + 5 extra databases) | run range (same) |
|---|---|---|---|---|---|---|---|---|
| v1 | 35.0% | 35.0% | 40.0% | 40.0% | 45.0% | 45.0% | 21.9–61.3% | 35–45% |
| v2 | 81.7% | 81.7% | 90.0% | 90.0% | 86.7% | 91.7% | 69.9–97.2% | 90–90% |
| v3 | 90.0% | 90.0% | 96.7% | 96.7% | 95.0% | 100.0% | 78.7–99.6% | 95–100% |

### Suite: holdout

| prompt | as recorded in the run | original grader, re-run | corrected grader | corrected grader + 5 extra databases | manual review, ambiguous = wrong | manual review, ambiguous = correct | 95% interval (corrected grader + 5 extra databases) | run range (same) |
|---|---|---|---|---|---|---|---|---|
| v1 | 50.0% | 50.0% | 60.0% | 60.0% | 63.3% | 63.3% | 38.7–78.1% | 60–60% |
| v2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 83.9–100.0% | 100–100% |
| v3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 83.9–100.0% | 100–100% |

### First-try accuracy (corrected grader + 5 extra databases)

| suite | v1 | v2 | v3 |
|---|---|---|---|
| dev | 40.0% | 90.0% | 96.7% |
| holdout | 58.3% | 100.0% | 100.0% |

## Agreement between the grader and the manual review

Answers reviewed by hand: **106** of 360 (every answer graded wrong, every answer whose verdict changed between graders, and a random sample of 20 graded correct). Labels and reasons: `evals/manual_review.json`.

| | answers |
|---|---|
| grader and reviewer agree | 95 |
| grader says wrong, reviewer says correct (false negative) | 5 |
| grader says correct, reviewer says wrong (false positive) | 0 |
| question allows two readings (ambiguous) | 6 |
| **agreement on unambiguous answers** | **95/100 (95.0%)** |

## 4. Paired comparison of prompt versions

Same questions, compared one by one (corrected grader + 5 extra databases). *Better / worse* count questions whose score (correct runs out of 3) went up or down. *p* is the exact two-sided sign test: the chance of a split at least this lopsided if the two versions were equally good.

| suite | comparison | better | worse | same | p |
|---|---|---|---|---|---|
| dev | v1 → v2 | 12 | 1 | 7 | 0.0034 |
| dev | v2 → v3 | 2 | 1 | 17 | 1.0000 |
| dev | v1 → v3 | 13 | 1 | 6 | 0.0018 |
| holdout | v1 → v2 | 9 | 0 | 11 | 0.0039 |
| holdout | v2 → v3 | 0 | 0 | 20 | 1.0000 |
| holdout | v1 → v3 | 9 | 0 | 11 | 0.0039 |

## 5. Per-question results

Correct runs out of 3 (corrected grader + 5 extra databases).

| suite | id | difficulty | v1 | v2 | v3 | question |
|---|---|---|---|---|---|---|
| dev | q01 | easy | 3/3 | 3/3 | 3/3 | How many customers do we have in each segment? |
| dev | q02 | easy | 3/3 | 3/3 | 3/3 | How many products are in each category? |
| dev | q03 | easy | 3/3 | 3/3 | 3/3 | How many orders were placed through each channel? |
| dev | q04 | easy | 0/3 | 3/3 | 3/3 | What is the total revenue for 2025? |
| dev | q05 | easy | 3/3 | 3/3 | 3/3 | How many orders were returned? |
| dev | q06 | medium | 0/3 | 3/3 | 3/3 | What is the total revenue by region? |
| dev | q07 | medium | 0/3 | 1/3 | 3/3 | What is the total revenue by product category? |
| dev | q08 | medium | 0/3 | 3/3 | 3/3 | Which 5 products generated the most revenue? |
| dev | q09 | medium | 0/3 | 3/3 | 3/3 | Show total revenue for each month of 2025 |
| dev | q10 | medium | 0/3 | 3/3 | 3/3 | What is the number of units sold per category? |
| dev | q11 | medium | 3/3 | 3/3 | 3/3 | How many completed orders did each region have? |
| dev | q12 | medium | 0/3 | 3/3 | 3/3 | What is the revenue by sales channel? |
| dev | q13 | medium | 1/3 | 3/3 | 3/3 | What is the return rate for each channel? |
| dev | q14 | hard | 0/3 | 3/3 | 3/3 | What was the revenue of the South region in Q3 2025? |
| dev | q15 | hard | 0/3 | 0/3 | 3/3 | What is the average order value for each customer segment? |
| dev | q16 | hard | 0/3 | 3/3 | 3/3 | What was total revenue in each quarter of 2025? |
| dev | q17 | hard | 2/3 | 3/3 | 3/3 | Compare Electronics revenue in the South region between Q2 and Q3 2025 |
| dev | q18 | hard | 0/3 | 3/3 | 3/3 | Who are the top 3 customers by revenue? Show their names. |
| dev | q19 | hard | 3/3 | 3/3 | 3/3 | What is the revenue by customer segment in the South region in Q3 2025? |
| dev | q20 | hard | 3/3 | 2/3 | 1/3 | What is the average discount given in each product category? |
| holdout | h01 | easy | 3/3 | 3/3 | 3/3 | How many different customers have at least one completed order in the Partner channel? |
| holdout | h02 | easy | 3/3 | 3/3 | 3/3 | How many customers does each region have? |
| holdout | h03 | easy | 3/3 | 3/3 | 3/3 | Which product has the highest list price, and what is that price? |
| holdout | h04 | easy | 3/3 | 3/3 | 3/3 | How many customers signed up in 2024? |
| holdout | h05 | easy | 3/3 | 3/3 | 3/3 | What is the cheapest list price in each product category? |
| holdout | h06 | medium | 3/3 | 3/3 | 3/3 | Which region had the most returned orders, and how many were there? |
| holdout | h07 | medium | 2/3 | 3/3 | 3/3 | How many units of each Software product were sold? |
| holdout | h08 | medium | 0/3 | 3/3 | 3/3 | How much revenue did Enterprise customers generate through each sales channel? |
| holdout | h09 | medium | 1/3 | 3/3 | 3/3 | How many completed orders were there in each quarter of 2025? |
| holdout | h10 | medium | 0/3 | 3/3 | 3/3 | Which 3 products sold the most units? Show the product names and units sold. |
| holdout | h11 | medium | 3/3 | 3/3 | 3/3 | On average, how many units are in a completed order? |
| holdout | h12 | medium | 0/3 | 3/3 | 3/3 | What was the revenue for each product category in December 2025? |
| holdout | h13 | medium | 3/3 | 3/3 | 3/3 | What percentage of orders were cancelled in each channel? |
| holdout | h14 | hard | 3/3 | 3/3 | 3/3 | Which single product category lost the most revenue between Q2 and Q3 2025? |
| holdout | h15 | hard | 0/3 | 3/3 | 3/3 | What is the average order value of Online orders compared with Retail orders? |
| holdout | h16 | hard | 0/3 | 3/3 | 3/3 | How much Electronics revenue did each region make in Q4 2025? |
| holdout | h17 | hard | 0/3 | 3/3 | 3/3 | Which month of 2025 had the highest revenue, and how much was it? |
| holdout | h18 | hard | 3/3 | 3/3 | 3/3 | How many customers have never returned an order? |
| holdout | h19 | hard | 0/3 | 3/3 | 3/3 | What percentage of total revenue came from each customer segment? |
| holdout | h20 | hard | 3/3 | 3/3 | 3/3 | What is the average line-item discount, as a percentage, on completed orders in each channel? |

## 6. Every verdict that changed

18 answers. *Gold* is the result of the hand-written query, *model* the result of the saved SQL (first rows shown).

- **dev run 1 v1 q20** (original → corrected grader): wrong → correct
  - question: What is the average discount given in each product category?
  - gold: `['category', 'avg_discount'] [('Electronics', 0.0529), ('Furniture', 0.05), ('Office Supplies', 0.049), ('Software', 0.0491)]`
  - model: `['category', 'average_discount'] [('Electronics', 0.0529), ('Furniture', 0.05), ('Office Supplies', 0.049), ('Software', 0.0491)]`
  - SQL: `SELECT p.category, AVG(oi.discount) AS average_discount FROM order_items AS oi JOIN products AS p ON oi.product_id = p.product_id GROUP BY p.category`
- **dev run 1 v2 q13** (original → corrected grader): wrong → correct
  - question: What is the return rate for each channel?
  - gold: `['channel', 'return_rate'] [('Online', 0.046), ('Partner', 0.0518), ('Retail', 0.0419)]`
  - model: `['channel', 'return_rate'] [('Online', 0.05), ('Partner', 0.05), ('Retail', 0.04)]`
  - SQL: `WITH order_counts AS ( SELECT o.channel, COUNT(*) AS total_orders, SUM(CASE WHEN o.status = 'returned' THEN 1 ELSE 0 END) AS returned_orders FROM orders AS o GROUP BY o.channel ) SELECT channel, ROUND(CAST(returned_orders AS REAL) / total_orders, 2) AS return_rate FROM order_counts ORDER BY channel LIMIT 500`
- **dev run 1 v2 q20** (original → corrected grader): wrong → correct
  - question: What is the average discount given in each product category?
  - gold: `['category', 'avg_discount'] [('Electronics', 0.0529), ('Furniture', 0.05), ('Office Supplies', 0.049), ('Software', 0.0491)]`
  - model: `['product_category', 'average_discount'] [('Software', 0.05), ('Office Supplies', 0.05), ('Furniture', 0.05), ('Electronics', 0.05)]`
  - SQL: `SELECT p.category AS product_category, ROUND(AVG(oi.discount), 2) AS average_discount FROM order_items AS oi JOIN orders AS o ON oi.order_id = o.order_id JOIN products AS p ON oi.product_id = p.product_id WHERE o.status = 'completed' GROUP BY p.category ORDER BY average_discount DESC LIMIT 500`
- **dev run 1 v3 q13** (original → corrected grader): wrong → correct
  - question: What is the return rate for each channel?
  - gold: `['channel', 'return_rate'] [('Online', 0.046), ('Partner', 0.0518), ('Retail', 0.0419)]`
  - model: `['channel', 'return_rate'] [('Partner', 0.05), ('Online', 0.05), ('Retail', 0.04)]`
  - SQL: `SELECT o.channel AS channel, ROUND(CAST(SUM(CASE WHEN o.status = 'returned' THEN 1 ELSE 0 END) AS REAL) / COUNT(*), 2) AS return_rate FROM orders o GROUP BY o.channel ORDER BY return_rate DESC LIMIT 500`
- **holdout run 1 v1 h11** (original → corrected grader): wrong → correct
  - question: On average, how many units are in a completed order?
  - gold: `['avg_units_per_order'] [(7.55,)]`
  - model: `['avg_units_per_completed_order'] [(7.5528,)]`
  - SQL: `SELECT AVG(qty_per_order) AS avg_units_per_completed_order FROM ( SELECT oi.order_id, SUM(oi.quantity) AS qty_per_order FROM order_items oi JOIN orders o ON oi.order_id = o.order_id WHERE o.status = 'completed' GROUP BY oi.order_id ) sub`
- **holdout run 1 v1 h20** (original → corrected grader): wrong → correct
  - question: What is the average line-item discount, as a percentage, on completed orders in each channel?
  - gold: `['channel', 'avg_discount_pct'] [('Online', 4.96), ('Partner', 5.31), ('Retail', 4.95)]`
  - model: `['channel', 'avg_discount_pct'] [('Online', 4.9616), ('Partner', 5.3097), ('Retail', 4.9471)]`
  - SQL: `SELECT o.channel, AVG(oi.discount) * 100.0 AS avg_discount_pct FROM orders o JOIN order_items oi ON o.order_id = oi.order_id WHERE o.status = 'completed' GROUP BY o.channel`
- **dev run 2 v1 q20** (original → corrected grader): wrong → correct
  - question: What is the average discount given in each product category?
  - gold: `['category', 'avg_discount'] [('Electronics', 0.0529), ('Furniture', 0.05), ('Office Supplies', 0.049), ('Software', 0.0491)]`
  - model: `['category', 'average_discount'] [('Electronics', 0.0529), ('Furniture', 0.05), ('Office Supplies', 0.049), ('Software', 0.0491)]`
  - SQL: `SELECT p.category, AVG(oi.discount) AS average_discount FROM order_items AS oi JOIN products AS p ON oi.product_id = p.product_id GROUP BY p.category`
- **dev run 2 v2 q13** (original → corrected grader): wrong → correct
  - question: What is the return rate for each channel?
  - gold: `['channel', 'return_rate'] [('Online', 0.046), ('Partner', 0.0518), ('Retail', 0.0419)]`
  - model: `['channel', 'return_rate'] [('Online', 0.05), ('Partner', 0.05), ('Retail', 0.04)]`
  - SQL: `WITH order_counts AS ( SELECT o.channel, COUNT(*) AS total_orders, SUM(CASE WHEN o.status = 'returned' THEN 1 ELSE 0 END) AS returned_orders FROM orders AS o GROUP BY o.channel ) SELECT channel, ROUND(CAST(returned_orders AS REAL) / NULLIF(total_orders, 0), 2) AS return_rate FROM order_counts ORDER BY channel LIMIT 500`
- **dev run 2 v3 q13** (original → corrected grader): wrong → correct
  - question: What is the return rate for each channel?
  - gold: `['channel', 'return_rate'] [('Online', 0.046), ('Partner', 0.0518), ('Retail', 0.0419)]`
  - model: `['channel', 'return_rate'] [('Online', 0.05), ('Partner', 0.05), ('Retail', 0.04)]`
  - SQL: `WITH order_counts AS ( SELECT o.channel AS channel, COUNT(*) AS total_orders, SUM(CASE WHEN o.status = 'returned' THEN 1 ELSE 0 END) AS returned_orders FROM orders o GROUP BY o.channel ) SELECT channel, ROUND(CASE WHEN total_orders = 0 THEN 0 ELSE (returned_orders * 1.0) / total_orders END, 2) AS return_rate FROM order_counts ORDER BY return_rate DESC LIMIT 500`
- **holdout run 2 v1 h11** (original → corrected grader): wrong → correct
  - question: On average, how many units are in a completed order?
  - gold: `['avg_units_per_order'] [(7.55,)]`
  - model: `['avg_units_per_completed_order'] [(7.5528,)]`
  - SQL: `SELECT AVG(qty_per_order) AS avg_units_per_completed_order FROM ( SELECT oi.order_id, SUM(oi.quantity) AS qty_per_order FROM order_items oi JOIN orders o ON oi.order_id = o.order_id WHERE o.status = 'completed' GROUP BY oi.order_id ) sub`
- **holdout run 2 v1 h20** (original → corrected grader): wrong → correct
  - question: What is the average line-item discount, as a percentage, on completed orders in each channel?
  - gold: `['channel', 'avg_discount_pct'] [('Online', 4.96), ('Partner', 5.31), ('Retail', 4.95)]`
  - model: `['channel', 'avg_discount_pct'] [('Online', 4.9616), ('Partner', 5.3097), ('Retail', 4.9471)]`
  - SQL: `SELECT o.channel, AVG(oi.discount) * 100.0 AS avg_discount_pct FROM orders o JOIN order_items oi ON o.order_id = oi.order_id WHERE o.status = 'completed' GROUP BY o.channel`
- **dev run 3 v1 q20** (original → corrected grader): wrong → correct
  - question: What is the average discount given in each product category?
  - gold: `['category', 'avg_discount'] [('Electronics', 0.0529), ('Furniture', 0.05), ('Office Supplies', 0.049), ('Software', 0.0491)]`
  - model: `['category', 'avg_discount'] [('Electronics', 0.0529), ('Furniture', 0.05), ('Office Supplies', 0.049), ('Software', 0.0491)]`
  - SQL: `SELECT p.category, AVG(oi.discount) AS avg_discount FROM order_items AS oi JOIN products AS p ON oi.product_id = p.product_id GROUP BY p.category`
- **dev run 3 v2 q13** (original → corrected grader): wrong → correct
  - question: What is the return rate for each channel?
  - gold: `['channel', 'return_rate'] [('Online', 0.046), ('Partner', 0.0518), ('Retail', 0.0419)]`
  - model: `['channel', 'return_rate'] [('Online', 0.05), ('Partner', 0.05), ('Retail', 0.04)]`
  - SQL: `WITH order_counts AS ( SELECT o.channel, COUNT(*) AS total_orders, SUM(CASE WHEN o.status = 'returned' THEN 1 ELSE 0 END) AS returned_orders FROM orders AS o GROUP BY o.channel ) SELECT oc.channel, ROUND(CAST(oc.returned_orders AS REAL) / oc.total_orders, 2) AS return_rate FROM order_counts AS oc ORDER BY oc.channel LIMIT 500`
- **dev run 3 v2 q20** (original → corrected grader): wrong → correct
  - question: What is the average discount given in each product category?
  - gold: `['category', 'avg_discount'] [('Electronics', 0.0529), ('Furniture', 0.05), ('Office Supplies', 0.049), ('Software', 0.0491)]`
  - model: `['product_category', 'avg_discount'] [('Software', 0.05), ('Office Supplies', 0.05), ('Furniture', 0.05), ('Electronics', 0.05)]`
  - SQL: `WITH completed_items AS ( SELECT oi.discount, p.category FROM order_items AS oi JOIN orders AS o ON oi.order_id = o.order_id JOIN products AS p ON oi.product_id = p.product_id WHERE o.status = 'completed' ) SELECT category AS product_category, ROUND(AVG(discount), 2) AS avg_discount FROM completed_items GROUP BY category ORDER BY avg_discount DESC LIMIT 500`
- **dev run 3 v3 q13** (original → corrected grader): wrong → correct
  - question: What is the return rate for each channel?
  - gold: `['channel', 'return_rate'] [('Online', 0.046), ('Partner', 0.0518), ('Retail', 0.0419)]`
  - model: `['channel', 'return_rate'] [('Partner', 0.05), ('Online', 0.05), ('Retail', 0.04)]`
  - SQL: `SELECT o.channel AS channel, ROUND(CAST(SUM(CASE WHEN o.status = 'returned' THEN 1 ELSE 0 END) AS REAL) / COUNT(*), 2) AS return_rate FROM orders o GROUP BY o.channel ORDER BY return_rate DESC LIMIT 500`
- **dev run 3 v3 q20** (original → corrected grader): wrong → correct
  - question: What is the average discount given in each product category?
  - gold: `['category', 'avg_discount'] [('Electronics', 0.0529), ('Furniture', 0.05), ('Office Supplies', 0.049), ('Software', 0.0491)]`
  - model: `['product_category', 'avg_discount'] [('Software', 0.05), ('Office Supplies', 0.05), ('Furniture', 0.05), ('Electronics', 0.05)]`
  - SQL: `WITH completed_items AS ( SELECT oi.discount, p.category FROM order_items oi JOIN orders o ON o.order_id = oi.order_id JOIN products p ON p.product_id = oi.product_id WHERE o.status = 'completed' ) SELECT category AS product_category, ROUND(AVG(discount), 2) AS avg_discount FROM completed_items GROUP BY category ORDER BY avg_discount DESC LIMIT 500`
- **holdout run 3 v1 h11** (original → corrected grader): wrong → correct
  - question: On average, how many units are in a completed order?
  - gold: `['avg_units_per_order'] [(7.55,)]`
  - model: `['avg_units_per_completed_order'] [(7.5528,)]`
  - SQL: `SELECT AVG(order_units) AS avg_units_per_completed_order FROM ( SELECT o.order_id, SUM(oi.quantity) AS order_units FROM orders o JOIN order_items oi ON o.order_id = oi.order_id WHERE o.status = 'completed' GROUP BY o.order_id ) sub`
- **holdout run 3 v1 h20** (original → corrected grader): wrong → correct
  - question: What is the average line-item discount, as a percentage, on completed orders in each channel?
  - gold: `['channel', 'avg_discount_pct'] [('Online', 4.96), ('Partner', 5.31), ('Retail', 4.95)]`
  - model: `['channel', 'avg_discount_pct'] [('Online', 4.9616), ('Partner', 5.3097), ('Retail', 4.9471)]`
  - SQL: `SELECT o.channel, AVG(oi.discount) * 100.0 AS avg_discount_pct FROM orders o JOIN order_items oi ON o.order_id = oi.order_id WHERE o.status = 'completed' GROUP BY o.channel`

