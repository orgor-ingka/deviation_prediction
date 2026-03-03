### Author: Niclas Lavesson
# This table contains the forecasted stock and volumes

CREATE OR REPLACE TABLE `ingka-ff-somdata-prod.OMDA_Analytics.forecasted_qty_volume` AS
WITH forecast AS 
(SELECT DISTINCT 
  f.local_date AS calendar_date,
  CONCAT(BUCurr.business_unit_type,'.', BUCurr.business_unit_code) AS business_unit,
  CONCAT(i.item_type || i.item_no) AS item_id,

  ROUND((operational_forecast),2) AS forecast_stock_current_day,
  ROUND((item_supply_chain_net_dm3/1000),2) AS forecast_volume_current_day,

  LAG(ROUND((operational_forecast),2)) OVER (PARTITION BY CONCAT(BUCurr.business_unit_type,'.', BUCurr.business_unit_code), CONCAT(i.item_type || i.item_no) ORDER BY f.local_date) AS forecast_stock_previous_day,
  LAG(ROUND((item_supply_chain_net_dm3/1000),2)) OVER (PARTITION BY CONCAT(BUCurr.business_unit_type,'.', BUCurr.business_unit_code), CONCAT(i.item_type || i.item_no) ORDER BY f.local_date) AS forecast_volume_previous_day,

  LEAD(ROUND((operational_forecast),2)) OVER (PARTITION BY CONCAT(BUCurr.business_unit_type,'.', BUCurr.business_unit_code), CONCAT(i.item_type || i.item_no) ORDER BY f.local_date) AS forecast_stock_next_day,
  LEAD(ROUND((item_supply_chain_net_dm3/1000),2)) OVER (PARTITION BY CONCAT(BUCurr.business_unit_type,'.', BUCurr.business_unit_code), CONCAT(i.item_type || i.item_no) ORDER BY f.local_date) AS forecast_volume_next_day

FROM
  `ingka-cff-insights-prod.dm_agg_inventory_no_pii.stock_inventory_day_agg_fct` f
  INNER JOIN `ingka-ilo-ia-prod.dm_ext_item.item_cur_dim` i ON f.item_ifk=i.item_isk
  INNER JOIN `ingka-ilo-ia-prod.dm_ext_business_unit.business_unit_cur_dim` AS BUCurr ON BUCurr.business_unit_isk=f.business_unit_ifk
  LEFT JOIN `ingka-ilo-ia-prod.dm_ext_range.item_range_fct` AS g ON g.item_ifk=f.item_ifk AND g.bu_ifk=f.business_unit_ifk AND f.local_date BETWEEN g.valid_from AND g.valid_to
  LEFT JOIN `ingka-ilo-ia-prod.dm_ext_range.unit_item_range_flag_dim` AS a ON a.fulfilment_unit_item_range_flag_isk=g.fulfilment_unit_item_range_flag_ifk  
  LEFT JOIN `ingka-cff-insights-prod.dm_agg_plan_demand_no_pii.daily_forecast_agg_fct` AS FC ON f.local_date = FC.local_forecast_date AND FC.business_unit_ifk=f.business_unit_ifk AND f.item_ifk=FC.item_ifk
  LEFT JOIN  `ingka-ilo-ia-prod.dm_ext_dwp.item_dwp_dim` AS Vol ON Vol.item_ifk = f.item_ifk AND f.local_date BETWEEN Vol.valid_from AND Vol.valid_to
WHERE (f.local_date BETWEEN '2025-09-30' AND '2025-12-31') 
      AND BUCurr.business_unit_type IN ('CDC') AND BUCurr.unit_open_status = 'Open'
      AND FC.local_forecast_date BETWEEN '2025-09-30' AND '2025-12-31'
  )
  SELECT *
  FROM forecast;