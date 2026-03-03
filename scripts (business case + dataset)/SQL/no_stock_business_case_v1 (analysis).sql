### Author: Niclas Lavesson
# This material is used in the business case for stock-related deviation predictions at CDCs

# Number of distinct orders
SELECT DISTINCT country_code, order_no
FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_business_case_deviations`;

# Number of distinct deliverires
SELECT DISTINCT country_code, order_no, parent_prime_line_no_DS
FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_business_case_deviations`;

# Number of distinct deliverires with a stock-related issue
SELECT DISTINCT country_code, order_no, parent_prime_line_no_DS
FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_business_case_deviations`
WHERE (stock_related_deviation_delivery = 1 OR unsuccessful_ar_delivery = 1)
AND successful_ar_delivery = 0

# Number of distinct deliverires with a deviation or an unsuccessful auto-recovery
SELECT DISTINCT country_code, order_no, prime_line_no
FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_business_case_deviations`
WHERE event_type = 'Deviation' OR ARF_type IN ('Case 5+6')

SELECT * 
FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_business_case_deviations`
WHERE (stock_related_deviation_delivery = 1 OR unsuccessful_ar_delivery = 1)
AND successful_ar_delivery = 0
ORDER BY country_code, order_no, parent_prime_line_no_DS

### Put this into the presentation:
-- 3 301 882 CDC orders in October 2025
-- 3 505 282 CDC deliveries in October 2025
  -- 24 064 of these deliveries had either a stock-related deviation or an unsuccessful auto-recovery due to inventory issue (but not a successful auto-recovery)
  -- 0.7% of CDC deliveries have a stock-related deviation issue

### Leave this aside
  ### All CDC-deliveries
    -- Total ordered qty: 36 825 204
    -- Total order value (euros): €949 791 274
    -- Avg ordered qty: 10.50
    -- Avg order value (euros): €271

  ### CDC-deliveries with stock-related issue
    -- Total ordered qty: 399 776 (1.1%)
    -- Total order value (euros): €11 487 396 (1.2%)
    -- Avg ordered qty: 16.61
    -- Avg order value (euros): €477

### For Orestis (October 2025 only)
  -- What percentage of total order lines deviates/have an unsuccessful auto-recovery? 
  -- Total orderlines we have + how many deviate

WITH tab0 AS (
SELECT *
FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_business_case_deviations`
QUALIFY ROW_NUMBER() OVER (PARTITION BY country_code, order_no, parent_prime_line_no_DS, prime_line_no) = 1),

tab1 AS (
SELECT country_code
      ,order_no
      ,parent_prime_line_no_DS
      ,SUM(ordered_qty) AS total_ordered_qty
      ,SUM(line_total_euros) AS total_euros
FROM tab0
GROUP BY country_code, order_no, parent_prime_line_no_DS)

SELECT SUM(total_ordered_qty) total_ordered_qty
      ,SUM(total_euros) total_euros
      ,AVG(total_ordered_qty) avg_ordered_qty
      ,AVG(total_euros) avg_euros
FROM tab1;

### CDC-deliveries with stock-related issue
WITH tab0 AS (
SELECT *
FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_business_case_deviations`
WHERE (stock_related_deviation_delivery = 1 OR unsuccessful_ar_delivery = 1)
AND successful_ar_delivery = 0
QUALIFY ROW_NUMBER() OVER (PARTITION BY country_code, order_no, parent_prime_line_no_DS, prime_line_no) = 1),

tab1 AS (
SELECT country_code
      ,order_no
      ,parent_prime_line_no_DS
      ,SUM(ordered_qty) AS total_ordered_qty
      ,SUM(line_total_euros) AS total_euros
FROM tab0
GROUP BY country_code, order_no, parent_prime_line_no_DS)

SELECT SUM(total_ordered_qty) total_ordered_qty
      ,SUM(total_euros) total_euros
      ,AVG(total_ordered_qty) avg_ordered_qty
      ,AVG(total_euros) avg_euros
FROM tab1;
