### Author: Niclas Lavesson
# This script creates the dataset used for the business case of the NO STOCK deviation predictions for CDCs.
# Parts of this dataset is replicated from the deviation predictions dataset described in the MIRO-board for this project: https://miro.com/app/board/uXjVJkYtT7k=/

 -- Step 1: Get product order attributes 
CREATE OR REPLACE TABLE `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_1_3m_v1` AS
WITH base_orders_prod_attributes AS (
  SELECT  
    RIGHT(sales.enterprisecode, 2) AS country_code,
    CAST(sales.orderno AS STRING) AS order_no,
    MIN(CASE WHEN TIMESTAMP(oo.StatusDate) < TIMESTAMP(sales.orderdate) THEN TIMESTAMP(oo.StatusDate) ELSE TIMESTAMP(sales.orderdate) END) OVER (PARTITION BY sales.enterprisecode, sales.orderno, oo.PrimeLineNo) AS order_date,
    sales.ordertype AS order_type,
    sales.extn.serviceType AS service_type,
    sales.entrytype AS sales_channel,
    oo.PrimeLineNo AS prime_line_no,
    oo.StatusDate AS line_status_date,
    oo.minLineStatus AS line_status_code,
    oo.minLineStatusDesc AS line_status_description, 
    oo.Item.ItemID AS item_id,
    oo.ShipNode AS ship_node,
    CAST(oo.orderedQty AS FLOAT64) AS ordered_qty,
    CAST(oo.linePriceInfo.RetailPrice AS FLOAT64) * CAST(oo.orderedQty AS FLOAT64) * CAST(b.exchg_rt AS FLOAT64) AS line_total_euros,
    oo.ItemDetails.Extn.TransportMethodType AS transport_method_type,
    oo.ItemGroupCode AS item_group_code,
    oo.Extn.LatestShipDate AS latest_dispatch_date,
    oo.Extn.WorkOrderNo AS work_order_no,
    oo.modifyts,
    MAX(CASE WHEN oo.minLineStatus IN ('9000','3700.01.100', '3700.02.100') THEN oo.StatusDate ELSE NULL END) OVER (PARTITION BY RIGHT(sales.enterprisecode, 2), CAST(sales.orderno AS STRING), oo.PrimeLineNo) AS cancellation_date,
    MAX(CASE WHEN oo.minLineStatus IN ('3700.7777','3700.110','3700.7777.010','3700.130','3200.140','3200.150','3700.7777.010') THEN oo.StatusDate ELSE NULL END) OVER (PARTITION BY RIGHT(sales.enterprisecode, 2), CAST(sales.orderno AS STRING), oo.PrimeLineNo) AS delivery_date

  FROM `ingka-ff-somdata-prod.order_management_no_pii.salesorder_trns_history` AS sales,
  UNNEST(orderlines.OrderLine) AS oo

LEFT JOIN (
  SELECT exchg_rt
    ,curcy_code_to
    ,curcy_code_from
  FROM `ingka-ff-somdata-prod.common_dim_no_pii.vw_currency_exchange_rate_dim`) AS b
  ON PriceInfo.currency = b.curcy_code_from
  AND b.curcy_code_to = 'EUR'

WHERE DATE(sales.orderDate) BETWEEN '2025-10-01' AND '2025-12-31'
  AND DATE(sales.modifyts) >= '2025-10-01' --AND DATE(sales.modifyts) <= '2025-10-15'
  AND DATE(sales.bq_ingestion_time) >= '2025-10-01' --AND DATE(sales.bq_ingestion_time) <= '2025-10-15'
  AND oo.ItemGroupCode = 'PROD'
 AND oo.Extn.LatestShipDate IS NOT NULL
QUALIFY ROW_NUMBER() OVER (PARTITION BY RIGHT(sales.enterprisecode, 2), CAST(sales.orderno AS STRING), oo.PrimeLineNo, oo.StatusDate, oo.minLineStatus ORDER BY 
oo.StatusDate DESC NULLS LAST, oo.modifyts DESC NULLS LAST, sales.ingestion_time DESC NULLS LAST, oo.Extn.LatestShipDate DESC NULLS LAST, oo.orderedQty DESC NULLS LAST, oo.linePriceInfo.RetailPrice DESC NULLS LAST) = 1
),

base_orders_prod_attributes12 AS (
  SELECT * FROM base_orders_prod_attributes
WHERE line_status_code NOT IN ('1300', '9000','3700.01.100', '3700.02.100')
),

base_orders_prod_attributes2 AS (
SELECT *
      ,ROW_NUMBER() OVER (PARTITION BY country_code, order_no, prime_line_no ORDER BY line_status_date, line_status_code) AS rn 
FROM base_orders_prod_attributes12
),

base_orders_prod_attributes3 AS (
SELECT * EXCEPT(modifyts)
FROM base_orders_prod_attributes2
QUALIFY ROW_NUMBER() OVER (PARTITION BY country_code, order_no, prime_line_no, line_status_description ORDER BY line_status_date DESC NULLS LAST, modifyts DESC NULLS LAST) = 1),

base_orders_prod_attributes4 AS (
SELECT * EXCEPT(rn, line_status_date)
      ,CASE
        WHEN line_status_description = 'Created' AND rn = 1 THEN TIMESTAMP(order_date) ELSE TIMESTAMP(line_status_date)
      END AS line_status_date
FROM base_orders_prod_attributes3
),

parent_prime_line_nos AS (
  SELECT 
    RIGHT(sales.enterprisecode, 2) AS country_code,
    CAST(sales.orderno AS STRING) AS order_no,
    oo.PrimeLineNo AS prime_line_no,
    CASE WHEN op.relationshipType = 'DELIVERY' THEN op.ParentLine.PrimeLineNo ELSE NULL END AS parent_prime_line_no_DS,
    CASE WHEN op.relationshipType = 'SERVICE' THEN op.ParentLine.PrimeLineNo ELSE NULL END AS parent_prime_line_no_PS,

  FROM `ingka-ff-somdata-prod.order_management_no_pii.salesorder_trns_history` AS sales,
  UNNEST(orderlines.OrderLine) AS oo
  LEFT JOIN UNNEST(oo.ParentOrderLineRelationships.OrderLineRelationship) AS op

WHERE DATE(sales.orderDate) BETWEEN '2025-10-01' AND '2025-12-31'
  AND DATE(sales.modifyts) >= '2025-10-01' --AND DATE(sales.modifyts) <= '2025-10-15'
  AND DATE(sales.bq_ingestion_time) >= '2025-10-01' --AND DATE(sales.bq_ingestion_time) <= '2025-10-15'
  AND oo.ItemGroupCode = 'PROD'
  AND op.relationshipType IS NOT NULL -- New (not run yet)
  AND op.ParentLine.PrimeLineNo IS NOT NULL
  AND oo.minLineStatus NOT IN ('1300', '9000','3700.01.100', '3700.02.100') --New (not run yet)
QUALIFY ROW_NUMBER() OVER (PARTITION BY RIGHT(sales.enterprisecode, 2), CAST(sales.orderno AS STRING), oo.PrimeLineNo, op.ParentLine.PrimeLineNo ORDER BY oo.StatusDate DESC NULLS LAST, oo.modifyts DESC NULLS LAST) = 1
),

base_orders_prod_attributes5 AS(
  SELECT a.*
        ,b.parent_prime_line_no_DS
        ,b.parent_prime_line_no_PS
  FROM base_orders_prod_attributes4 AS a
  LEFT JOIN parent_prime_line_nos AS b
    ON a.country_code = b.country_code
    AND a.order_no = b.order_no
    AND a.prime_line_no = b.prime_line_no
),

-- Step 2a: Get delivery service line attributes
base_orders_delivery_service_attributes AS (
SELECT 
    RIGHT(sales.enterprisecode, 2) AS country_code,
    CAST(sales.orderno AS STRING) AS order_no,
    oo.PrimeLineNo AS prime_line_no_DS,
    oo.Item.ItemID AS level_of_service_DS,
    oo.PromisedApptStartDate AS promised_appt_start_date_DS,
    oo.PromisedApptEndDate AS promised_appt_end_date_DS
  FROM `ingka-ff-somdata-prod.order_management_no_pii.salesorder_trns_history` AS sales,
  UNNEST(orderlines.OrderLine) AS oo
  LEFT JOIN UNNEST(oo.ParentOrderLineRelationships.OrderLineRelationship) AS op
  LEFT JOIN UNNEST(sales.paymentmethods.PaymentMethod) AS pm

WHERE DATE(sales.orderDate) BETWEEN '2025-10-01' AND '2025-12-31'
  AND DATE(sales.modifyts) >= '2025-10-01' -- AND DATE(sales.modifyts) <= '2025-10-15'
  AND DATE(sales.bq_ingestion_time) >= '2025-10-01' --AND DATE(sales.bq_ingestion_time) <= '2025-10-15'
  AND oo.ItemGroupCode IN ('DS')
  AND oo.minLineStatus NOT IN ('1300', '9000','3700.01.100', '3700.02.100') --New (not run yet)
QUALIFY ROW_NUMBER() OVER (PARTITION BY RIGHT(sales.enterprisecode, 2), CAST(sales.orderno AS STRING), oo.PrimeLineNo ORDER BY oo.StatusDate DESC NULLS LAST, oo.modifyts DESC NULLS LAST, oo.extn.PromisedApptStartDate DESC NULLS LAST, oo.extn.PromisedApptEndDate DESC NULLS LAST) = 1),

-- Step 2b: Get provided service line attributes
base_orders_provided_service_attributes AS (
SELECT 
    RIGHT(sales.enterprisecode, 2) AS country_code,
    CAST(sales.orderno AS STRING) AS order_no,
    oo.PrimeLineNo AS prime_line_no_PS,
    oo.Item.ItemID AS level_of_service_PS,
    sales.ingestion_time,
    oo.PromisedApptStartDate AS promised_appt_start_date_PS,
    oo.PromisedApptEndDate AS promised_appt_end_date_PS
  FROM `ingka-ff-somdata-prod.order_management_no_pii.salesorder_trns_history` AS sales,
  UNNEST(orderlines.OrderLine) AS oo
  LEFT JOIN UNNEST(oo.ParentOrderLineRelationships.OrderLineRelationship) AS op
  LEFT JOIN UNNEST(sales.paymentmethods.PaymentMethod) AS pm

WHERE DATE(sales.orderDate) BETWEEN '2025-10-01' AND '2025-12-31'
  AND DATE(sales.modifyts) >= '2025-10-01' --AND DATE(sales.modifyts) <= '2025-10-15'
  AND DATE(sales.bq_ingestion_time) >= '2025-10-01' --AND DATE(sales.bq_ingestion_time) <= '2025-10-15'
  AND oo.ItemGroupCode IN ('PS')
  AND oo.minLineStatus NOT IN ('1300', '9000','3700.01.100', '3700.02.100') --New (not run yet)
QUALIFY ROW_NUMBER() OVER (PARTITION BY RIGHT(sales.enterprisecode, 2), CAST(sales.orderno AS STRING), oo.PrimeLineNo ORDER BY oo.StatusDate DESC NULLS LAST, oo.modifyts DESC NULLS LAST, oo.extn.PromisedApptStartDate DESC NULLS LAST, oo.extn.PromisedApptEndDate DESC NULLS LAST) = 1),

-- Step 2c: Get WO holds
WO_holds AS (
      SELECT RIGHT(sales.enterprisecode,2) AS country_code
            ,CAST(sales.orderno AS STRING) AS order_no
            ,ol.primeLineNo AS prime_line_no
            ,oo.ParentLine.PrimeLineNo AS parent_prime_line_no
            ,oht.status AS oht_status
            ,oht.createts AS createts
            ,oht.modifyts  AS modifyts
            FROM `ingka-ff-somdata-prod.order_management_no_pii.salesorder_trns_history` AS sales,
            UNNEST(orderlines.OrderLine) AS ol,
            UNNEST(ol.ChildOrderLineRelationships.OrderLineRelationship) AS oo,
            UNNEST(ol.orderHoldTypes.orderHoldType) AS oht

            WHERE DATE(sales.orderDate) BETWEEN '2025-10-01' AND '2025-12-31'
              AND DATE(sales.modifyts) >= '2025-10-01' --AND DATE(sales.modifyts) <= '2025-10-15'
              AND DATE(sales.bq_ingestion_time) >= '2025-10-01' --AND DATE(sales.bq_ingestion_time) <= '2025-10-15'

            AND oht.holdType = 'WOConfHold'
            AND oht.status IN ('1100','1300')
      QUALIFY ROW_NUMBER() OVER (PARTITION BY RIGHT(sales.enterprisecode,2), sales.orderno, ol.primeLineNo, oht_status ORDER BY oht.createts ASC NULLS LAST, oht.modifyts ASC NULLS LAST) = 1),
 
WO_hold_create AS (
      SELECT country_code
            ,order_no
            ,parent_prime_line_no
            ,modifyts AS WO_creation_date
FROM WO_holds
      WHERE oht_status = '1100'),
 
WO_hold_modify AS (
      SELECT country_code
            ,order_no
            ,parent_prime_line_no
            ,modifyts AS WO_acknowledgement_date
FROM WO_holds
      WHERE oht_status = '1300'),

-- Step 2d: Get order payment data
payment_orders AS
(SELECT country_code
        ,CAST(order_no AS STRING) AS order_no 
        ,created_at AS created_date
        ,payment_type
        ,payment_due_at
FROM `ingka-sp-spdp-prod.sales_orders.sales_orders_payment_transactions_events_pii` 
WHERE DATE(updated_at) >= '2025-10-01'
AND DATE(created_at) >= '2025-10-01'
AND payment_due_at IS NOT NULL
QUALIFY ROW_NUMBER() OVER (PARTITION BY CAST(order_no AS STRING) ORDER BY created_at NULLS LAST, updated_at NULLS LAST, payment_due_at ASC NULLS LAST) = 1),

fully_paid_orders AS
(SELECT country_code
      ,CAST(order_no AS STRING) AS order_no 
      ,updated_at AS payment_fully_paid_date
FROM `ingka-sp-spdp-prod.sales_orders.sales_orders_payment_transactions_events_pii` 
WHERE DATE(updated_at) >= '2025-10-01'
AND DATE(created_at) >= '2025-10-01'
AND payment_status = 'FULLY_PAID'
QUALIFY ROW_NUMBER() OVER (PARTITION BY CAST(order_no AS STRING) ORDER BY created_at NULLS LAST, updated_at ASC NULLS LAST) = 1),

-- Step 2e: Get processing times
processing_times AS
(SELECT a.createts
      ,a.modifyts
      ,a.data_region
      ,a.node_notification_key
      ,a.node_notification_perd_key
      ,a.min_notification_time
      ,a.level_of_service
      ,b.organization_code
FROM `ingka-ff-somdata-prod.common_dim.node_notification_dim` AS a
LEFT JOIN `ingka-ff-somdata-prod.common_dim.node_notification_perd_dim` AS b
ON a.node_notification_perd_key = b.node_notification_perd_key
WHERE DATE(a.createts) < CURRENT_DATE()
AND a.min_notification_time != 0
QUALIFY ROW_NUMBER() OVER (PARTITION BY level_of_service, organization_code ORDER BY createts DESC NULLS LAST, a.modifyts DESC NULLS LAST) = 1),

-- Step 2f: Get customer details (Private/Business/NA)
customer_details AS
(SELECT retail_unit_code AS country_code
      ,CAST(order_no AS STRING) AS order_no
      ,customer_type
FROM `ingka-sbp-di-prod.common_sales_mart_no_pii.common_sales_bd`
WHERE DATE(transaction_start_ts) >= '2025-10-01' AND DATE(transaction_start_ts) < '2026-01-01'
AND DATE(transaction_date) >= '2025-10-01'
QUALIFY ROW_NUMBER() OVER (PARTITION BY retail_unit_code, order_no ORDER BY transaction_start_ts DESC NULLS LAST, transaction_end_ts DESC NULLS LAST) = 1),

-- Step 3: Join data from step 2a-2f
merged_sales_orders_attributes AS (
SELECT a.*
      ,COUNT(DISTINCT a.parent_prime_line_no_DS) OVER (PARTITION BY a.country_code, a.order_no) AS no_of_DS
      ,COUNT(DISTINCT a.parent_prime_line_no_PS) OVER (PARTITION BY a.country_code, a.order_no) AS no_of_PS
      ,b.* EXCEPT(country_code, order_no, prime_line_no_DS)
      ,c.* EXCEPT(country_code, order_no, prime_line_no_PS)

      ,d.WO_creation_date
      ,e.WO_acknowledgement_date
      ,f.payment_type
      ,f.payment_due_at
      ,g.payment_fully_paid_date
      ,h.min_notification_time
      ,i.customer_type

      ,CASE
        WHEN ds.promised_update_DS IS NOT NULL THEN ds.promised_update_DS-1 ELSE 0
      END AS promised_update_DS

      ,CASE
        WHEN ps.promised_update_PS IS NOT NULL THEN ps.promised_update_PS-1 ELSE 0 
      END AS promised_update_PS

FROM base_orders_prod_attributes5 AS a

  LEFT JOIN base_orders_delivery_service_attributes AS b
  ON a.order_no = b.order_no
  AND a.country_code = b.country_code
  AND a.parent_prime_line_no_DS = b.prime_line_no_DS

  LEFT JOIN base_orders_provided_service_attributes AS c
  ON a.order_no = c.order_no
  AND a.country_code = c.country_code
  AND a.parent_prime_line_no_PS = c.prime_line_no_PS

  LEFT JOIN WO_hold_create AS d
    ON a.country_code = d.country_code
    AND a.order_no = d.order_no
    AND a.parent_prime_line_no_DS = d.parent_prime_line_no

  LEFT JOIN WO_hold_modify AS e
    ON a.country_code = e.country_code
    AND a.order_no = e.order_no
    AND a.parent_prime_line_no_DS = e.parent_prime_line_no

  LEFT JOIN payment_orders AS f
    ON a.country_code = f.country_code
    AND a.order_no = f.order_no

  LEFT JOIN fully_paid_orders AS g
    ON a.country_code = g.country_code
    AND a.order_no = g.order_no

  LEFT JOIN processing_times AS h
    ON b.level_of_service_DS = h.level_of_service
    AND a.ship_node = h.organization_code

  LEFT JOIN customer_details AS i
    ON a.country_code = i.country_code
    AND a.order_no = i.order_no
    
  LEFT JOIN (
      SELECT RIGHT(enterprise_code,2) AS country_code
            ,order_no
            ,prime_line_no AS prime_line_no_DS
            ,COUNT(*) AS promised_update_DS
      FROM `ingka-ff-somdata-prod.data_mart.vw_so_promised_delivery_ts_hist`
      WHERE DATE(order_date) BETWEEN '2025-10-01' AND '2025-12-31'
      AND line_type = 'DS'
      GROUP BY country_code, order_no, prime_line_no) AS ds
    ON a.country_code = ds.country_code AND a.order_no = ds.order_no AND a.parent_prime_line_no_DS = ds.prime_line_no_DS

  LEFT JOIN (
      SELECT RIGHT(enterprise_code,2) AS country_code
            ,order_no
            ,prime_line_no AS prime_line_no_PS
            ,COUNT(*) AS promised_update_PS
      FROM `ingka-ff-somdata-prod.data_mart.vw_so_promised_delivery_ts_hist`
      WHERE DATE(order_date) BETWEEN '2025-10-01' AND '2025-12-31'
      AND line_type = 'PS'
      GROUP BY country_code, order_no, prime_line_no) AS ps
    ON a.country_code = ps.country_code AND a.order_no = ps.order_no AND a.parent_prime_line_no_PS = ps.prime_line_no_PS
),

merged_sales_orders_attributes2 AS (
  SELECT * EXCEPT(promised_appt_start_date_DS, promised_appt_end_date_DS, promised_appt_start_date_PS, promised_appt_end_date_PS, WO_creation_date, WO_acknowledgement_date, payment_fully_paid_date)
  ,MAX(promised_appt_start_date_DS) OVER (PARTITION BY country_code, order_no, prime_line_no) AS promised_appt_start_date_DS
  ,MAX(promised_appt_end_date_DS) OVER (PARTITION BY country_code, order_no, prime_line_no) AS promised_appt_end_date_DS
  ,MAX(promised_appt_start_date_PS) OVER (PARTITION BY country_code, order_no, prime_line_no) AS promised_appt_start_date_PS
  ,MAX(promised_appt_end_date_PS) OVER (PARTITION BY country_code, order_no, prime_line_no) AS promised_appt_end_date_PS
  ,MAX(WO_creation_date) OVER (PARTITION BY country_code, order_no, prime_line_no) AS WO_creation_date
  ,MAX(WO_acknowledgement_date) OVER (PARTITION BY country_code, order_no, prime_line_no) AS WO_acknowledgement_date
  ,MAX(payment_fully_paid_date) OVER (PARTITION BY country_code, order_no, prime_line_no) AS payment_fully_paid_date
  FROM merged_sales_orders_attributes
  WHERE LEFT(ship_node,3) = 'CDC'
  QUALIFY ROW_NUMBER() OVER (PARTITION BY country_code, order_no, prime_line_no, line_status_description ORDER BY line_status_date)=1),

merged_sales_orders_attributes3 AS (
SELECT *
      ,LEAD(line_status_date) OVER (PARTITION BY country_code, order_no, prime_line_no ORDER BY line_status_date, line_status_code) AS first_next_line_status_date
FROM merged_sales_orders_attributes2),

merged_sales_orders_attributes4 AS (
SELECT *
      ,LEAD(line_status_date) OVER (PARTITION BY country_code, order_no, prime_line_no ORDER BY line_status_date, line_status_code) AS next_line_status_date
FROM merged_sales_orders_attributes3
WHERE line_status_description IN ('Created', 'Allocated', 'Sent for fulfillment')
),

-- Step 4: Fix next line status date
merged_sales_orders_attributes5 AS (
SELECT * EXCEPT(first_next_line_status_date, next_line_status_date)
      ,CASE
        WHEN line_status_description IN ('Created', 'Allocated') AND TIMESTAMP(next_line_status_date) IS NOT NULL THEN TIMESTAMP(next_line_status_date)
        WHEN line_status_description IN ('Created', 'Allocated') AND TIMESTAMP(next_line_status_date) IS NULL AND TIMESTAMP(cancellation_date) IS NOT NULL THEN TIMESTAMP(cancellation_date)
        WHEN line_status_description = 'Sent for fulfillment' AND TIMESTAMP(first_next_line_status_date) IS NOT NULL THEN TIMESTAMP(first_next_line_status_date)
        WHEN line_status_description = 'Sent for fulfillment' AND TIMESTAMP(first_next_line_status_date) IS NULL AND TIMESTAMP(cancellation_date) IS NOT NULL THEN TIMESTAMP(cancellation_date)
        ELSE CURRENT_TIMESTAMP()
      END AS next_line_status_date
FROM merged_sales_orders_attributes4
WHERE line_status_description IN ('Created', 'Allocated', 'Sent for fulfillment')
)

SELECT * FROM merged_sales_orders_attributes5;


-- Step 5: Add minimum notification dates
CREATE OR REPLACE TABLE `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_11_3m_v1` AS
WITH calendar AS (
SELECT * EXCEPT(shift_end)
      ,CASE
        WHEN shift_end = '23:59:00' THEN '23:59:59' -- We need to add milliseconds here : Since otherwise there is a risk that we miss these orders
        ELSE shift_end
      END AS shift_end
      ,CEIL(TIME_DIFF(shift_end, shift_start, SECOND) / 3600.0) AS capacity_in_hours
FROM `ingka-ff-somdata-prod.OMDA_Analytics.unit_operation_hours_2025`),

calendar2 AS(
SELECT * EXCEPT(shift_start, shift_end)
      ,DATETIME(calendar_date, shift_start) AS shift_start_date
      ,DATETIME(calendar_date, shift_end) AS shift_end_date
FROM calendar),

tab1 AS(
SELECT DISTINCT a.country_code
      ,a.order_no
      ,a.prime_line_no
      ,a.line_status_code
      ,DATETIME(a.order_date, 'Europe/Paris') AS order_date
      ,a.ship_node
      ,DATETIME(a.latest_dispatch_date, 'Europe/Paris') AS latest_dispatch_date
      ,a.min_notification_time
      ,b.* EXCEPT(organization_code)
FROM (SELECT * FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_1_3m_v1`
      WHERE min_notification_time IS NOT NULL AND latest_dispatch_date IS NOT NULL) AS a
LEFT JOIN calendar2 AS b
  ON a.ship_node = b.organization_code
  AND DATE_SUB(DATE(order_date), INTERVAL 28 DAY) <= calendar_date
  AND calendar_date <= DATE(latest_dispatch_date)),

tab2 AS (
  SELECT *
       ,CASE
          WHEN DATE(latest_dispatch_date) != DATE(calendar_date) THEN capacity_in_hours
          
          WHEN latest_dispatch_date < shift_start_date THEN 0
          
          WHEN latest_dispatch_date BETWEEN shift_start_date AND shift_end_date 
          THEN ROUND(TIMESTAMP_DIFF(CAST(latest_dispatch_date AS TIMESTAMP), CAST(shift_start_date AS TIMESTAMP), SECOND) / 3600.0,3)
          
          ELSE capacity_in_hours
        END AS capacity_in_hours_actual
  FROM tab1),

tab3 AS(
SELECT *
      ,SUM(capacity_in_hours_actual) OVER 
            (PARTITION BY order_no, country_code, prime_line_no, line_status_code 
            ORDER BY calendar_date DESC) 
      AS cumulative_capacity_in_hours
FROM tab2),
 
tab4 AS(
SELECT *
      ,(cumulative_capacity_in_hours - min_notification_time) AS excess_processing_time
FROM tab3),

tab5 AS (
SELECT *
      ,CASE 
            WHEN DATE(latest_dispatch_date) > calendar_date AND excess_processing_time < 0
            THEN NULL

            WHEN DATE(latest_dispatch_date) = calendar_date AND excess_processing_time >= 0
            THEN TIMESTAMP_SUB(latest_dispatch_date, INTERVAL CAST(min_notification_time AS INT64) HOUR)

            WHEN DATE(latest_dispatch_date) > calendar_date AND excess_processing_time >= 0
            THEN TIMESTAMP_ADD(shift_start_date, INTERVAL CAST(excess_processing_time * 60 AS INT64) MINUTE)

      END AS min_notification_date
FROM tab4),

tab6 AS
(SELECT *
FROM tab5
WHERE excess_processing_time >= 0
QUALIFY ROW_NUMBER() OVER (PARTITION BY country_code, order_no, prime_line_no, line_status_code ORDER BY calendar_date DESC NULLS LAST) = 1),

tab7 AS
(SELECT * EXCEPT(min_notification_date)
      ,TIMESTAMP(min_notification_date, 'Europe/Paris') AS min_notification_date
FROM tab6),

-- Add ISOM inventory table here
-- Add forecast table here

merged_sales_orders_attributes6 AS (
SELECT a.*
      ,b.min_notification_date
      ,c.* EXCEPT(calendar_date, business_unit, item_id)
FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_1_3m_v1` AS a
LEFT JOIN tab7 AS b
  ON a.country_code = b.country_code AND a.order_no = b.order_no AND a.prime_line_no = b.prime_line_no AND a.line_status_code = b.line_status_code
LEFT JOIN `ingka-ff-somdata-prod.OMDA_Analytics.forecasted_qty_volume` AS c
  ON DATE(a.line_status_date) = DATE(c.calendar_date)
  AND a.ship_node = c.business_unit
  AND a.item_id = c.item_id)

SELECT * FROM merged_sales_orders_attributes6;


-- Adding deviations + HFB + DDC item + SL item
CREATE OR REPLACE TABLE `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_business_case_deviations` AS
WITH get_deviations AS (
SELECT *
FROM `ingka-ff-somdata-prod.OMDA_Analytics.ODM_deviations_and_auto_recoveries_2024onwards` 
WHERE (
    -- (deviation_reason_code = 'NO STOCK' AND deviation_occurence IN ('ORDER PROCESSING', 'FULFILMENT')
    --   OR 
      (deviation_reason_code = 'DELAY CAUSED BY IKEA' AND deviation_occurence = "FULFILMENT" AND reason_caused = "No Stock")
      OR (deviation_reason_code IN ('MANUAL DATE SET','NO SUPPLY','QUAL BLOCK','SALESSTOP – NOSUPPLY')))
AND DATE(order_date) BETWEEN '2025-10-01' AND '2025-12-31'
AND DATE(event_time)>= '2025-10-01'
AND event_type = 'Deviation' 
OR (event_type = 'Auto-recovery' AND auto_recovered_line IS NOT NULL)
),

get_no_stock_deviations AS (
SELECT country_code
      ,order_no
      ,prime_line_no
      ,event_time
      ,1 AS has_no_stock_deviation
FROM `ingka-ff-somdata-prod.OMDA_Analytics.ODM_deviations_and_auto_recoveries_2024onwards` 
WHERE (deviation_reason_code = 'NO STOCK' AND deviation_occurence IN ('ORDER PROCESSING', 'FULFILMENT'))),

joining_data AS (
SELECT a.*
      ,b.event_type
      ,b.event_time 
      ,b.deviation_reason_code
      ,b.deviation_occurence
      ,b.reason_caused
      ,b.ARF_type
      ,CASE
        WHEN c.has_no_stock_deviation IS NOT NULL THEN 1 ELSE 0
      END AS has_no_stock_deviation
FROM `ingka-ff-somdata-prod.OMDA_Analytics.no_stock_deviation_predictions_11_3m_v1` AS a
LEFT JOIN get_deviations AS b
  ON a.country_code = b.country_code
  AND a.order_no = b.order_no
  AND a.prime_line_no = b.prime_line_no
  AND ((line_status_date <= b.event_time) AND (b.event_time) < next_line_status_date)
LEFT JOIN get_no_stock_deviations AS c
  ON a.country_code = c.country_code
  AND a.order_no = c.order_no
  AND a.prime_line_no = c.prime_line_no
  AND ((line_status_date <= c.event_time) AND (c.event_time) < next_line_status_date)),

joining_data2 AS (
SELECT *
      ,MAX(CASE WHEN event_type = 'Deviation' THEN 1 ELSE 0 END) OVER (PARTITION BY country_code, order_no, parent_prime_line_no_DS) AS stock_related_deviation_delivery
      ,MAX(CASE WHEN event_type = 'Auto-recovery' THEN 1 ELSE 0 END) OVER (PARTITION BY country_code, order_no, parent_prime_line_no_DS) AS ar_delivery
      ,MAX(CASE WHEN ARF_type IN ('Case 1', 'Case 2', 'Case 3', 'Case 4') THEN 1 ELSE 0 END) OVER (PARTITION BY country_code, order_no, parent_prime_line_no_DS) AS successful_ar_delivery
      ,MAX(CASE WHEN ARF_type IN ('Case 5+6') THEN 1 ELSE 0 END) OVER (PARTITION BY country_code, order_no, parent_prime_line_no_DS) AS unsuccessful_ar_delivery
FROM joining_data)

SELECT *
FROM joining_data2
WHERE DATE(order_date) BETWEEN '2025-10-01' AND '2025-10-31';


