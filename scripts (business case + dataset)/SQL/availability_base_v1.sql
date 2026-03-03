# ATP/ISOM Availability  
 
# Description: This analysis aims to leverage the ATP/ISOM availability data in order to come up with variables which can be used in the prediction of kitchen order deviation predictions
  
# Query created on: 23rd of January, 2025
# Query created by: Christopher Brewer and Niclas Lavesson
# First, we are creating a table from the `ingka-ofd-atpdata-prod.atp_derived_layer.item_availability` dataset

CREATE OR REPLACE TABLE `ingka-ff-somdata-prod.OMDA_Analytics.availability_base`
PARTITION BY DATE(modify_date_time) CLUSTER BY item_id, business_unit AS
SELECT country_code
      ,item_id
      ,business_unit
      ,modify_date_time
      ,availability_qty
FROM `ingka-ofd-atpdata-prod.atp_derived_layer.item_availability`
WHERE DATE(modify_date_time) >= '2024-01-01'
QUALIFY ROW_NUMBER() OVER (PARTITION BY item_id, business_unit, modify_date_time, CAST(availability_qty AS STRING)) = 1;

# Creating a new table, where we are adding all the new fields to the base inventory dataset
--DROP TABLE `ingka-ff-somdata-prod.OMDA_Analytics.availability_full`;
CREATE OR REPLACE TABLE `ingka-ff-somdata-prod.OMDA_Analytics.availability_full` 
PARTITION BY DATE(modify_date_time) CLUSTER BY item_id, business_unit AS

WITH tab1 AS
(SELECT country_code,
  business_unit,
  item_id,
  modify_date_time,
  availability_qty,
      # average stock (28 days)
  AVG(availability_qty) OVER (
    PARTITION BY business_unit, item_id
    ORDER BY UNIX_SECONDS(modify_date_time)
    RANGE BETWEEN 2419200 PRECEDING AND CURRENT ROW
  ) AS average_stock_4_weeks,
      # stock variance (28 days)
  VARIANCE(availability_qty) OVER (
    PARTITION BY business_unit, item_id
    ORDER BY UNIX_SECONDS(modify_date_time)
    RANGE BETWEEN 2419200 PRECEDING AND CURRENT ROW
  ) AS stock_variance_4_weeks
FROM `ingka-ff-somdata-prod.OMDA_Analytics.availability_base`),

inventory_dates AS
(SELECT item_id
      ,business_unit
      ,DATE(modify_date_time) AS modify_date
      ,MIN(availability_qty) AS min_availability_qty
FROM
  `ingka-ff-somdata-prod.OMDA_Analytics.availability_base`
GROUP BY item_id
      ,business_unit
      ,DATE(modify_date_time)),

inventory_dates_2 AS
(SELECT *
      ,LAG(min_availability_qty) OVER (PARTITION BY business_unit, item_id ORDER BY modify_date ASC) AS lag_availability_qty
      ,LAG(modify_date) OVER (PARTITION BY business_unit, item_id ORDER BY modify_date ASC) AS lag_modify_date
FROM inventory_dates),

inventory_dates_3 AS
(SELECT *
      ,DATE_DIFF(modify_date, lag_modify_date, DAY) AS date_difference
FROM inventory_dates_2),

inventory_dates_4 AS
(SELECT *
      ,(min_availability_qty-lag_availability_qty) AS availability_difference
      # current_daily_stock_change_rate
      ,(min_availability_qty-lag_availability_qty)/date_difference AS current_daily_stock_change_rate
FROM inventory_dates_3),

replenishment_averages_table AS
(SELECT item_id
      ,business_unit
      ,modify_date
      # average_daily_stock_change_rate_replenishment
      ,AVG(current_daily_stock_change_rate) OVER (
    PARTITION BY business_unit, item_id
    ORDER BY UNIX_DATE(modify_date)
    RANGE BETWEEN 28 PRECEDING AND CURRENT ROW
) AS average_daily_stock_change_rate_replenishment
FROM inventory_dates_4),

no_replenishment_averages_table AS(
SELECT item_id
      ,business_unit
      ,modify_date
      # average_daily_stock_change_rate
      ,AVG(current_daily_stock_change_rate) OVER (
    PARTITION BY business_unit, item_id
    ORDER BY UNIX_DATE(modify_date)
    RANGE BETWEEN 28 PRECEDING AND CURRENT ROW
) AS average_daily_stock_change_rate
FROM inventory_dates_4
WHERE current_daily_stock_change_rate <= 0
),

both_rates AS
(SELECT a.*
        ,b.average_daily_stock_change_rate_replenishment
        ,c.average_daily_stock_change_rate
FROM inventory_dates_4 AS a
LEFT JOIN replenishment_averages_table AS b
ON a.business_unit = b.business_unit AND a.item_id = b.item_id AND a.modify_date = b.modify_date
LEFT JOIN no_replenishment_averages_table AS c
ON a.business_unit = c.business_unit AND a.item_id = c.item_id AND a.modify_date = c.modify_date),

tab2 AS
(SELECT a.*
      ,b.current_daily_stock_change_rate
      ,b.average_daily_stock_change_rate_replenishment
      ,b.average_daily_stock_change_rate
FROM tab1 AS a
LEFT JOIN both_rates AS b
ON a.business_unit = b.business_unit AND a.item_id = b.item_id AND DATE(a.modify_date_time) = b.modify_date),

lag_availability_tab AS
(SELECT *
      ,DATE(modify_date_time) AS modify_date
      ,LAG(availability_qty) OVER (PARTITION BY business_unit, item_id ORDER BY modify_date_time ASC) AS lag_availability_qty
FROM `ingka-ff-somdata-prod.OMDA_Analytics.availability_base`),

stockout_tab AS
(SELECT *
      ,CASE
            WHEN availability_qty = 0 AND lag_availability_qty > 0 THEN 1
            ELSE 0
      END AS stockout_dummy
FROM lag_availability_tab),

stockout_tab_2 AS
(SELECT *
      # stockout_count_28_days
      ,SUM(stockout_dummy) OVER (
      PARTITION BY business_unit, item_id
      ORDER BY UNIX_SECONDS(modify_date_time)
      RANGE BETWEEN 2419200 PRECEDING AND CURRENT ROW) AS stockout_count_28_days
FROM stockout_tab),

stockout_only AS
(SELECT * FROM stockout_tab_2
WHERE stockout_dummy = 1),

stockout_only_2 AS
(SELECT * EXCEPT(modify_date_time)
      ,modify_date_time AS stockout_date_time
      ,LEAD(modify_date_time) OVER (PARTITION BY item_id, business_unit ORDER BY modify_date_time) AS lead_stockout_datetime
FROM stockout_only),

stockout_only_3 AS
(SELECT * EXCEPT(lead_stockout_datetime)
      ,CASE 
            WHEN lead_stockout_datetime IS NULL THEN CURRENT_TIMESTAMP()
            ELSE lead_stockout_datetime
      END AS lead_stockout_datetime
FROM stockout_only_2),

stockout_tab_3 AS
(SELECT a.*
      ,b.stockout_date_time
      ,DATE(b.stockout_date_time) AS stockout_date
FROM stockout_tab_2 AS a
LEFT JOIN stockout_only_3 AS b
ON a.item_id = b.item_id AND a.business_unit = b.business_unit AND a.modify_date_time >= b.stockout_date_time AND a.modify_date_time < b.lead_stockout_datetime
),

stockout_tab_4 AS
(SELECT * 
      # days_since_last_stockout
      ,DATE_DIFF(modify_date, stockout_date, DAY) AS days_since_last_stockout
FROM stockout_tab_3),

replenishment_tab AS
(SELECT *
      ,CASE
            WHEN availability_qty > lag_availability_qty THEN 1
            ELSE 0
      END AS replenishment_dummy
FROM lag_availability_tab),

replenishment_tab_2 AS
(SELECT *
      # replenishment_count_28_days
      ,SUM(replenishment_dummy) OVER (
      PARTITION BY business_unit, item_id
      ORDER BY UNIX_SECONDS(modify_date_time)
      RANGE BETWEEN 2419200 PRECEDING AND CURRENT ROW) AS replenishment_count_28_days
FROM replenishment_tab),

replenishment_only AS
(SELECT * FROM replenishment_tab_2
WHERE replenishment_dummy = 1),

replenishment_only_2 AS
(SELECT * EXCEPT(modify_date_time)
      ,modify_date_time AS replenishment_date_time
      ,LEAD(modify_date_time) OVER (PARTITION BY item_id, business_unit ORDER BY modify_date_time) AS lead_replenishment_datetime
FROM replenishment_only),

replenishment_only_3 AS
(SELECT * EXCEPT(lead_replenishment_datetime)
      ,CASE 
            WHEN lead_replenishment_datetime IS NULL THEN CURRENT_TIMESTAMP()
            ELSE lead_replenishment_datetime
      END AS lead_replenishment_datetime
FROM replenishment_only_2),

replenishment_tab_3 AS
(SELECT a.*
      ,b.replenishment_date_time
      ,DATE(b.replenishment_date_time) AS replenishment_date
FROM replenishment_tab_2 AS a
LEFT JOIN replenishment_only_3 AS b
ON a.item_id = b.item_id AND a.business_unit = b.business_unit AND a.modify_date_time >= b.replenishment_date_time AND a.modify_date_time < b.lead_replenishment_datetime
),

replenishment_tab_4 AS
(SELECT * 
      # days_since_last_stockout
      ,DATE_DIFF(modify_date, replenishment_date, DAY) AS days_since_last_replenishment
FROM replenishment_tab_3),

# Adding all the new fields to the base dataset

tab3 AS
(SELECT a.*
      ,b.stockout_count_28_days
      ,b.days_since_last_stockout
      ,c.replenishment_count_28_days
      ,c.days_since_last_replenishment
FROM tab2 AS a
LEFT JOIN stockout_tab_4 AS b
ON a.item_id = b.item_id AND a.business_unit = b.business_unit AND a.modify_date_time = b.modify_date_time AND a.availability_qty = b.availability_qty
LEFT JOIN replenishment_tab_4 AS c
ON a.item_id = c.item_id AND a.business_unit = c.business_unit AND a.modify_date_time = c.modify_date_time AND a.availability_qty = c.availability_qty
)

# Finally, we are going to add the following fields:
-- estimated_days_till_next_stock out

SELECT *
      ,CASE
            WHEN availability_qty = 0 THEN 0
            WHEN ABS(average_daily_stock_change_rate) = 0 AND ABS(current_daily_stock_change_rate) !=0 
            THEN FLOOR(availability_qty/ABS(current_daily_stock_change_rate))
            WHEN ABS(average_daily_stock_change_rate) != 0 
            THEN FLOOR(availability_qty/ABS(average_daily_stock_change_rate))
            WHEN average_daily_stock_change_rate IS NULL AND average_daily_stock_change_rate_replenishment != 0 
            THEN FLOOR(availability_qty/ABS(average_daily_stock_change_rate_replenishment))
            ELSE NULL
      END AS estimated_days_till_next_stockout
FROM tab3
WHERE modify_date_time >= '2025-10-01'
-- We need to sort by available_qty (take the smallest value) - because there are many duplicates
QUALIFY ROW_NUMBER() OVER (PARTITION BY business_unit, item_id, modify_date_time ORDER BY availability_qty ASC) = 1;


SELECT COUNT(*)
SELECT *
FROM `ingka-ff-somdata-prod.OMDA_Analytics.availability_full` 
LIMIT 1000
WHERE days_since_last_stockout IS NULL


