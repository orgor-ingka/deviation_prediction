### Author: Niclas Lavesson
# The dataset used is from the BQ-script "no_stock_deviation_predictions_3months_v1"
# These datasets are described in the MIRO-board for this project: https://miro.com/app/board/uXjVJkYtT7k=/

CREATE OR REPLACE TABLE `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_dataset_3months_v1` AS
WITH tab1 AS (
SELECT * EXCEPT(promised_appt_end_date_PS, promised_appt_start_date_PS, ingestion_time, level_of_service_PS, parent_prime_line_no_PS, parent_prime_line_no_DS, promised_appt_start_date_DS, promised_appt_end_date_DS)

      ,SUM(ordered_qty) OVER (PARTITION BY country_code, order_no, parent_prime_line_no_DS) AS ordered_qty_delivery
      ,SUM(ordered_qty) OVER (PARTITION BY country_code, order_no) AS ordered_qty_order

      ,SUM(CASE WHEN ddc_item = 1 THEN ordered_qty ELSE 0 END) OVER (PARTITION BY country_code, order_no, prime_line_no) AS ddc_ordered_qty
      ,SUM(CASE WHEN ddc_item = 1 THEN ordered_qty ELSE 0 END) OVER (PARTITION BY country_code, order_no,parent_prime_line_no_DS) AS ddc_ordered_qty_delivery
      ,SUM(CASE WHEN ddc_item = 1 THEN ordered_qty ELSE 0 END) OVER (PARTITION BY country_code, order_no) AS ddc_ordered_qty_order

      ,SUM(CASE WHEN item_id_service_level = '1' THEN ordered_qty ELSE 0 END) OVER (PARTITION BY country_code, order_no, prime_line_no) AS sl1_ordered_qty
      ,SUM(CASE WHEN item_id_service_level = '2' THEN ordered_qty ELSE 0 END) OVER (PARTITION BY country_code, order_no, prime_line_no) AS sl2_ordered_qty
      ,SUM(CASE WHEN item_id_service_level = '3' THEN ordered_qty ELSE 0 END) OVER (PARTITION BY country_code, order_no, prime_line_no) AS sl3_ordered_qty
      ,SUM(CASE WHEN item_id_service_level = '4' THEN ordered_qty ELSE 0 END) OVER (PARTITION BY country_code, order_no, prime_line_no) AS sl4_ordered_qty

      ,SUM(CASE WHEN item_id_service_level = '1' THEN ordered_qty ELSE 0 END) OVER (PARTITION BY country_code, order_no, parent_prime_line_no_DS) AS sl1_ordered_qty_delivery
      ,SUM(CASE WHEN item_id_service_level = '2' THEN ordered_qty ELSE 0 END) OVER (PARTITION BY country_code, order_no, parent_prime_line_no_DS) AS sl2_ordered_qty_delivery
      ,SUM(CASE WHEN item_id_service_level = '3' THEN ordered_qty ELSE 0 END) OVER (PARTITION BY country_code, order_no, parent_prime_line_no_DS) AS sl3_ordered_qty_delivery
      ,SUM(CASE WHEN item_id_service_level = '4' THEN ordered_qty ELSE 0 END) OVER (PARTITION BY country_code, order_no, parent_prime_line_no_DS) AS sl4_ordered_qty_delivery

      ,SUM(CASE WHEN item_id_service_level = '1' THEN ordered_qty ELSE 0 END) OVER (PARTITION BY country_code, order_no) AS sl1_ordered_qty_order
      ,SUM(CASE WHEN item_id_service_level = '2' THEN ordered_qty ELSE 0 END) OVER (PARTITION BY country_code, order_no) AS sl2_ordered_qty_order
      ,SUM(CASE WHEN item_id_service_level = '3' THEN ordered_qty ELSE 0 END) OVER (PARTITION BY country_code, order_no) AS sl3_ordered_qty_order
      ,SUM(CASE WHEN item_id_service_level = '4' THEN ordered_qty ELSE 0 END) OVER (PARTITION BY country_code, order_no) AS sl4_ordered_qty_order
      ,SUM(CASE WHEN item_id_service_level IS NULL THEN ordered_qty ELSE 0 END) OVER (PARTITION BY country_code, order_no) AS no_sl_ordered_qty_order

      ,CASE
        WHEN item_id_service_level = '1' THEN 1 ELSE 0
      END AS sl1_item

      ,CASE
        WHEN item_id_service_level = '2' THEN 1 ELSE 0
      END AS sl2_item

      ,CASE
        WHEN item_id_service_level = '3' THEN 1 ELSE 0
      END AS sl3_item

      ,CASE
        WHEN item_id_service_level = '4' THEN 1 ELSE 0
      END AS sl4_item

      ,CASE
        WHEN item_id_service_level IS NULL THEN 1 ELSE 0
      END AS no_sl_item

  -- time in status in hours
      ,TIMESTAMP_DIFF(next_line_status_date,line_status_date,SECOND)/60 AS time_in_status_mins
      ,TIMESTAMP_DIFF(next_line_status_date,line_status_date,SECOND)/3600 AS time_in_status_hours
      ,TIMESTAMP_DIFF(next_line_status_date, line_status_date, SECOND) / 86400 AS time_in_status_days

  -- time to wo_creation_date in hours
      ,TIMESTAMP_DIFF(WO_creation_date,line_status_date,SECOND)/60 AS time_to_wo_creation_date_mins
      ,TIMESTAMP_DIFF(WO_creation_date,line_status_date,SECOND)/3600 AS time_to_wo_creation_date_hours
      ,TIMESTAMP_DIFF(WO_creation_date,line_status_date,SECOND)/86400 AS time_to_wo_creation_date_days

  -- time to WO_acknowledgement_date in hours
      ,TIMESTAMP_DIFF(WO_acknowledgement_date,line_status_date,SECOND)/60 AS time_to_WO_acknowledgement_date_mins
      ,TIMESTAMP_DIFF(WO_acknowledgement_date,line_status_date,SECOND)/3600 AS time_to_WO_acknowledgement_date_hours
      ,TIMESTAMP_DIFF(WO_acknowledgement_date,line_status_date,SECOND)/86400 AS time_to_WO_acknowledgement_date_days

  -- time to min_notification_date in hours
      ,TIMESTAMP_DIFF(min_notification_date,line_status_date,SECOND)/60 AS time_to_min_notification_date_mins
      ,TIMESTAMP_DIFF(min_notification_date,line_status_date,SECOND)/3600 AS time_to_min_notification_date_hours
      ,TIMESTAMP_DIFF(min_notification_date,line_status_date,SECOND)/86400 AS time_to_min_notification_date_days

  -- time to promised_delivery (lead time variability - option 1)
      ,TIMESTAMP_DIFF(promised_appt_start_date_DS,line_status_date,SECOND)/60 AS time_to_promised_appt_start_date_mins
      ,TIMESTAMP_DIFF(promised_appt_start_date_DS,line_status_date,SECOND)/3600 AS time_to_promised_appt_start_date_hours
      ,TIMESTAMP_DIFF(promised_appt_start_date_DS,line_status_date,SECOND)/86400 AS time_to_promised_appt_start_date_days

  -- time to promised_delivery (lead time variability - option 2)
      ,TIMESTAMP_DIFF(latest_dispatch_date,line_status_date,SECOND)/60 AS time_to_dispatch_date_mins
      ,TIMESTAMP_DIFF(latest_dispatch_date,line_status_date,SECOND)/3600 AS time_to_dispatch_date_hours
      ,TIMESTAMP_DIFF(latest_dispatch_date,line_status_date,SECOND)/86400 AS time_to_dispatch_date_days

    ,CASE
      WHEN event_type IS NOT NULL THEN 1 ELSE 0
    END AS deviation

FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_raw_3m_v1`
),

dev_tab1 AS (
  SELECT country_code
        ,order_no
        ,prime_line_no
        ,ship_node
        ,deviation_reason_code
        ,deviation_occurence
        ,reason_caused
        ,DATE(event_time) AS event_date
  FROM `ingka-ff-somdata-prod.OMDA_Analytics.ODM_deviations_and_auto_recoveries_2024onwards` 
  WHERE DATE(order_date) BETWEEN '2025-09-01' AND '2025-12-31'
    --AND DATE(event_time) BETWEEN '2025-09-01' AND '2025-12-31'
    AND item_group_code != 'DS'
),

dev_calendar AS (
  SELECT calendar_date
  FROM UNNEST(GENERATE_DATE_ARRAY('2025-10-01', '2026-01-31', INTERVAL 1 DAY)) AS calendar_date
),

-- Getting total orders
sales_orders_omdm2 AS (
  SELECT DATE(order_date) AS order_date
        ,ship_node
        ,COUNT(DISTINCT(CONCAT(country_code, order_no, prime_line_no))) AS number_of_orders
  FROM `ingka-ff-somdata-prod.omdm_2_no_pii.sales_order_line`
  WHERE DATE(order_date) BETWEEN '2025-09-01' AND '2025-12-31'
    AND item_group_code = 'PROD'
  GROUP BY order_date, ship_node
),

sales_orders_rolling_1month AS (
  SELECT s1.order_date
        ,s1.ship_node
        ,SUM(s2.number_of_orders) AS number_of_orders_1month
  FROM sales_orders_omdm2 s1
  LEFT JOIN sales_orders_omdm2 s2
    ON s2.ship_node = s1.ship_node
    AND s2.order_date BETWEEN DATE_SUB(s1.order_date, INTERVAL 1 MONTH) AND s1.order_date
  GROUP BY s1.order_date, s1.ship_node
),

dev_tab2 AS (
  SELECT c.calendar_date
        ,t.ship_node
        ,COUNT(DISTINCT(CONCAT(country_code, order_no, prime_line_no))) AS number_of_deviations
        ,COUNT(CASE 
                 WHEN (t.deviation_reason_code = 'NO STOCK' AND t.deviation_occurence IN ('ORDER PROCESSING', 'FULFILMENT'))
                   OR (t.deviation_reason_code = 'DELAY CAUSED BY IKEA' AND t.deviation_occurence = 'FULFILMENT' AND t.reason_caused = 'No Stock')
                   OR (t.deviation_reason_code IN ('MANUAL DATE SET','NO SUPPLY','QUAL BLOCK','SALESSTOP – NOSUPPLY')) 
                 THEN 1 
                 ELSE NULL 
               END) AS number_of_stock_related_deviations
  FROM dev_calendar c
  LEFT JOIN dev_tab1 t 
    ON DATE(t.event_date) BETWEEN DATE_SUB(c.calendar_date, INTERVAL 1 MONTH) AND DATE_SUB(c.calendar_date, INTERVAL 1 DAY)
  GROUP BY c.calendar_date, t.ship_node
),

final_tab AS (
SELECT a.*
      ,100*(b.number_of_deviations/s.number_of_orders_1month) AS ship_node_deviation_rate_1month
      ,100*(b.number_of_stock_related_deviations/s.number_of_orders_1month) AS ship_node_stock_related_deviation_rate_1month
FROM tab1 AS a
LEFT JOIN dev_tab2 AS b
  ON a.ship_node = b.ship_node
  AND DATE(a.order_date) = b.calendar_date

LEFT JOIN sales_orders_rolling_1month s
  ON DATE(s.order_date) = DATE_SUB(b.calendar_date, INTERVAL 1 DAY)
  AND s.ship_node = a.ship_node)

SELECT *
FROM final_tab
ORDER BY country_code, order_no, prime_line_no, line_status_date, event_time;


### Below different samples used by DS when doing predictive modeling is created:

### Create 10% sample
--DROP TABLE `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_dataset_v1_10_sample`;
-- CREATE OR REPLACE TABLE `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_dataset_v1_10_sample`
-- PARTITION BY DATE(order_date) AS 
-- SELECT *
-- FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_dataset_v1`
-- WHERE MOD(ABS(FARM_FINGERPRINT(CAST(CONCAT(country_code, order_no) AS STRING))), 10) = 0;


-- ### Keep all deviations + three times as many zeros (ratio 3:1)
-- CREATE OR REPLACE TABLE `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_dataset_ratio_3to1` AS
-- WITH Deviation1Rows AS (
--     SELECT * FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_dataset_v1`
--     WHERE deviation = 1),

--   CountDeviation1 AS (
--     SELECT COUNT(*) AS num_deviation_1_rows FROM Deviation1Rows),

--   Deviation0Ranked AS (
--     SELECT t.*,
--       ROW_NUMBER() OVER (ORDER BY FARM_FINGERPRINT(TO_JSON_STRING(t))) AS rn
--       FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_dataset_v1` AS t
--       WHERE t.deviation = 0),

--   Deviation0Sample AS (
--     SELECT
--       t.* EXCEPT (rn) -- Exkludera den temporära radnummerkolumnen
--       FROM Deviation0Ranked AS t, CountDeviation1 AS c
--       WHERE t.rn <= (c.num_deviation_1_rows * 3))

-- SELECT * FROM Deviation1Rows
-- UNION ALL
-- SELECT * FROM Deviation0Sample;


-- -- For created status
-- CREATE OR REPLACE TABLE `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_dataset_ratio_3to1_created` AS
-- WITH Deviation1Rows AS (
--     SELECT * FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_dataset_v1`
--   WHERE line_status_description = 'Created'
--   AND deviation = 1),

--   CountDeviation1 AS (
--     SELECT COUNT(*) AS num_deviation_1_rows FROM Deviation1Rows
-- ),

--   Deviation0Ranked AS (
--     SELECT t.*,
--       ROW_NUMBER() OVER (ORDER BY FARM_FINGERPRINT(TO_JSON_STRING(t))) AS rn
--       FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_dataset_v1` AS t
--       WHERE t.deviation = 0
--       AND line_status_description = 'Created'),

--   Deviation0Sample AS (
--     SELECT
--       t.* EXCEPT (rn) -- Exkludera den temporära radnummerkolumnen
--       FROM Deviation0Ranked AS t, CountDeviation1 AS c
--       WHERE t.rn <= (c.num_deviation_1_rows * 3))

-- SELECT * FROM Deviation1Rows
-- UNION ALL
-- SELECT * FROM Deviation0Sample;


-- -- For allocated status
-- CREATE OR REPLACE TABLE `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_dataset_ratio_3to1_allocated` AS
-- WITH Deviation1Rows AS (
--     SELECT * FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_dataset_v1`
--     WHERE line_status_description = 'Allocated'
--     AND deviation = 1),

--   CountDeviation1 AS (
--     SELECT COUNT(*) AS num_deviation_1_rows FROM Deviation1Rows),

--   Deviation0Ranked AS (
--     SELECT t.*,
--       ROW_NUMBER() OVER (ORDER BY FARM_FINGERPRINT(TO_JSON_STRING(t))) AS rn
--       FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_dataset_v1` AS t
--       WHERE t.deviation = 0
--       AND line_status_description = 'Allocated'),

--   Deviation0Sample AS (
--     SELECT
--       t.* EXCEPT (rn) -- Exkludera den temporära radnummerkolumnen
--       FROM Deviation0Ranked AS t, CountDeviation1 AS c
--       WHERE t.rn <= (c.num_deviation_1_rows * 3))

-- SELECT * FROM Deviation1Rows
-- UNION ALL
-- SELECT * FROM Deviation0Sample;


-- -- For sff status
-- CREATE OR REPLACE TABLE `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_dataset_ratio_3to1_sff` AS
-- WITH Deviation1Rows AS (
--     SELECT * FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_dataset_v1`
--     WHERE line_status_description = 'Sent for fulfillment'
--     AND deviation = 1),

--   CountDeviation1 AS (
--     SELECT COUNT(*) AS num_deviation_1_rows FROM Deviation1Rows),

--   Deviation0Ranked AS (
--     SELECT t.*,
--       ROW_NUMBER() OVER (ORDER BY FARM_FINGERPRINT(TO_JSON_STRING(t))) AS rn
--       FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_dataset_v1` AS t
--       WHERE t.deviation = 0
--       AND line_status_description = 'Sent for fulfillment'),

--   Deviation0Sample AS (
--     SELECT
--       t.* EXCEPT (rn) -- Exkludera den temporära radnummerkolumnen
--       FROM Deviation0Ranked AS t, CountDeviation1 AS c
--       WHERE t.rn <= (c.num_deviation_1_rows * 3))

-- SELECT * FROM Deviation1Rows
-- UNION ALL
-- SELECT * FROM Deviation0Sample;

