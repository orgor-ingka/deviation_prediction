### Author: Niclas Lavesson
# This material is used in the business case for stock-related deviation predictions at CDCs

# Create a new table used for cancellation analysis below
CREATE OR REPLACE TABLE `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_cancellation_analysis_v1` AS
WITH tab0 AS(
SELECT country_code
      ,order_no
      ,parent_prime_line_no_DS
      ,prime_line_no
      ,item_id
      ,cancellation_date
      ,CASE WHEN cancellation_date IS NOT NULL THEN 1 ELSE 0
      END AS cancelled
      ,stock_related_deviation_delivery
      ,ar_delivery
      ,successful_ar_delivery
      ,unsuccessful_ar_delivery
      ,line_total_euros
FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_business_case_deviations`
WHERE parent_prime_line_no_DS IS NOT NULL
QUALIFY ROW_NUMBER() OVER (PARTITION BY country_code, order_no, parent_prime_line_no_DS, prime_line_no ORDER BY line_status_code) = 1),

tab1 AS (
SELECT country_code, order_no, parent_prime_line_no_DS
      ,STRING_AGG(item_id,', ' ORDER BY item_id) AS item_ids
      ,MAX(cancelled) AS cancelled
      ,MAX(stock_related_deviation_delivery) stock_related_deviation_delivery
      ,MAX(ar_delivery) ar_delivery
      ,MAX(successful_ar_delivery) successful_ar_delivery
      ,MAX(unsuccessful_ar_delivery) unsuccessful_ar_delivery
      ,SUM(line_total_euros) AS total_euros
FROM tab0
GROUP BY country_code, order_no, parent_prime_line_no_DS),

tab2 AS (
SELECT * 
  ,LAST_VALUE(cancelled) OVER (
    PARTITION BY country_code, order_no, item_ids
    ORDER BY CAST(parent_prime_line_no_DS AS INT64) ASC  -- Sortera stigande
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING  -- ← KRITISKT!
  ) AS last_cancelled_for_items
FROM tab1)

SELECT *
FROM tab2
ORDER BY country_code, order_no, CAST(parent_prime_line_no_DS AS INT64)

### Cancellation values (for deviations and unsuccessful auto-recoveries)
SELECT DISTINCT country_code, order_no, parent_prime_line_no_DS
FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_cancellation_analysis_v1`
WHERE (stock_related_deviation_delivery = 1 OR unsuccessful_ar_delivery = 1)
AND successful_ar_delivery = 0
AND last_cancelled_for_items = 1

-- Total deliveries 23 402
-- Total cancelled deliveries 16 051
-- 68,6% Cancellation rate

### Cancellation values (for successful auto-recoveries)
SELECT DISTINCT country_code, order_no, parent_prime_line_no_DS
FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_cancellation_analysis_v1`
WHERE successful_ar_delivery = 1
AND last_cancelled_for_items = 1

### Cancellation values
SELECT SUM(total_euros)
FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_cancellation_analysis_v1`
WHERE (stock_related_deviation_delivery = 1 OR unsuccessful_ar_delivery = 1)
AND successful_ar_delivery = 1
AND last_cancelled_for_items = 1

-- Total deliveries 1747
-- Total cancelled deliveries 3871
-- 45,1% Cancellation rate

