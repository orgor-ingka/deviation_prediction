### Author: Niclas Lavesson
# This script creates the dataset used for availability with imputed values. This is used when running the auto-recovery analysis logit regression models.

CREATE OR REPLACE TABLE `ingka-ff-somdata-prod.OMDA_Analytics.availability_modified_base`
PARTITION BY DATE(modify_date_time) CLUSTER BY item_id, business_unit AS
SELECT a.item_id
      ,a.business_unit
      ,a.modify_date_time
      ,a.availability_qty
      ,b.country_code
FROM `ingka-ofd-atpdata-prod.atp_derived_layer.item_availability` AS a
LEFT JOIN (
      SELECT DISTINCT country_code
      ,ship_node
      FROM `ingka-ff-somdata-prod.omdm_2_no_pii.sales_order_line`
      WHERE LEFT(ship_node, 3) = 'CDC') AS b
ON a.business_unit = b.ship_node
WHERE DATE(modify_date_time) >= '2025-06-01'
AND LEFT(business_unit,3) = 'CDC'
QUALIFY ROW_NUMBER() OVER (PARTITION BY item_id, business_unit, modify_date_time, CAST(availability_qty AS STRING)) = 1;

# Creating a new table, where we are adding all the new fields to the base inventory dataset
DROP TABLE `ingka-ff-somdata-prod.OMDA_Analytics.availability_modified`;
CREATE OR REPLACE TABLE `ingka-ff-somdata-prod.OMDA_Analytics.availability_modified` AS

WITH daily_snapshots AS (
  SELECT 
    country_code,
    business_unit,
    item_id,
    DATE(modify_date_time) as snapshot_date,
    MAX(modify_date_time) as last_update_time,
    ARRAY_AGG(availability_qty ORDER BY modify_date_time DESC LIMIT 1)[OFFSET(0)] 
      as end_of_day_availability_qty
  FROM `ingka-ff-somdata-prod.OMDA_Analytics.availability_modified_base`
  GROUP BY country_code, business_unit, item_id, DATE(modify_date_time)
),

rolling_30d_stats AS (
  SELECT 
    country_code,
    business_unit,
    item_id,
    snapshot_date,
    end_of_day_availability_qty,
    
    AVG(end_of_day_availability_qty) OVER (
      PARTITION BY country_code, business_unit, item_id 
      ORDER BY snapshot_date 
      ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
    ) as avg_30d_availability,
    
    STDDEV(end_of_day_availability_qty) OVER (
      PARTITION BY country_code, business_unit, item_id 
      ORDER BY snapshot_date 
      ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
    ) as stddev_30d_availability,
    
    COUNT(*) OVER (
      PARTITION BY country_code, business_unit, item_id 
      ORDER BY snapshot_date 
      ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
    ) as n_days_in_window
    
  FROM daily_snapshots
),

-- === IMPUTATION LOGIC ===
-- Beräkna fallback values för items utan tillräcklig historik
imputation_baselines AS (
  SELECT 
    country_code,
    business_unit,
    item_id,
    
    -- Globalt median z-score per business unit + item (över alla dagar)
    APPROX_QUANTILES(
      CASE 
        WHEN stddev_30d_availability > 0 AND n_days_in_window >= 20
        THEN (end_of_day_availability_qty - avg_30d_availability) / stddev_30d_availability
      END, 
      100
    )[OFFSET(50)] as median_item_z_score,
    
    -- Antal gånger item HAR haft z-score
    COUNTIF(stddev_30d_availability > 0 AND n_days_in_window >= 20) as n_valid_z_scores
    
  FROM rolling_30d_stats
  GROUP BY country_code, business_unit, item_id
),

-- Backup: Business unit level median (om item-level saknas)
business_unit_baseline AS (
  SELECT 
    country_code,
    business_unit,
    
    APPROX_QUANTILES(
      CASE 
        WHEN stddev_30d_availability > 0 AND n_days_in_window >= 20
        THEN (end_of_day_availability_qty - avg_30d_availability) / stddev_30d_availability
      END, 
      100
    )[OFFSET(50)] as median_bu_z_score
    
  FROM rolling_30d_stats
  GROUP BY country_code, business_unit
)

SELECT 
  rs.country_code,
  rs.business_unit,
  rs.item_id,
  rs.snapshot_date,
  rs.end_of_day_availability_qty as current_availability,
  rs.avg_30d_availability,
  rs.stddev_30d_availability,
  rs.n_days_in_window,
  
  -- === ORIGINAL Z-SCORE (kan vara NULL) ===
  CASE 
    WHEN rs.stddev_30d_availability > 0 AND rs.n_days_in_window >= 20
    THEN (rs.end_of_day_availability_qty - rs.avg_30d_availability) / rs.stddev_30d_availability
    ELSE NULL
  END as stock_z_score_original,
  
  -- === IMPUTED Z-SCORE (alltid har värde) ===
  COALESCE(
    -- 1. Använd original z-score om available
    CASE 
      WHEN rs.stddev_30d_availability > 0 AND rs.n_days_in_window >= 20
      THEN (rs.end_of_day_availability_qty - rs.avg_30d_availability) / rs.stddev_30d_availability
    END,
    
    -- 2. Fallback: Item's median z-score (om item har data andra dagar)
    CASE WHEN ib.n_valid_z_scores >= 5 THEN ib.median_item_z_score END,
    
    -- 3. Fallback: Business unit median z-score
    bu.median_bu_z_score,
    
    -- 4. Final fallback: 0 (neutral)
    0
    
  ) as stock_z_score_imputed,
  
  -- === IMPUTATION FLAG ===
  CASE 
    WHEN rs.stddev_30d_availability > 0 AND rs.n_days_in_window >= 20 THEN 'OBSERVED'
    WHEN ib.n_valid_z_scores >= 5 THEN 'IMPUTED_ITEM_MEDIAN'
    WHEN bu.median_bu_z_score IS NOT NULL THEN 'IMPUTED_BU_MEDIAN'
    ELSE 'IMPUTED_NEUTRAL'
  END as imputation_method,
  
  CASE 
    WHEN rs.stddev_30d_availability > 0 AND rs.n_days_in_window >= 20 THEN 0
    ELSE 1
  END as stock_z_score_imputed_flag,
  
  -- === BINARY INDICATORS (using imputed values) ===
  CASE 
    WHEN COALESCE(
      CASE WHEN rs.stddev_30d_availability > 0 AND rs.n_days_in_window >= 20
        THEN (rs.end_of_day_availability_qty - rs.avg_30d_availability) / rs.stddev_30d_availability
      END,
      CASE WHEN ib.n_valid_z_scores >= 5 THEN ib.median_item_z_score END,
      bu.median_bu_z_score,
      0
    ) < 0 THEN 1 ELSE 0 
  END as below_avg_imputed,
  
  CASE 
    WHEN COALESCE(
      CASE WHEN rs.stddev_30d_availability > 0 AND rs.n_days_in_window >= 20
        THEN (rs.end_of_day_availability_qty - rs.avg_30d_availability) / rs.stddev_30d_availability
      END,
      CASE WHEN ib.n_valid_z_scores >= 5 THEN ib.median_item_z_score END,
      bu.median_bu_z_score,
      0
    ) < -1 THEN 1 ELSE 0 
  END as stock_1sd_below_imputed,
  
  -- === ORIGINAL BINARY (kan vara NULL) ===
  CASE WHEN rs.end_of_day_availability_qty < rs.avg_30d_availability THEN 1 ELSE 0 END 
    as below_30d_avg_original

FROM rolling_30d_stats rs
LEFT JOIN imputation_baselines ib 
  ON rs.country_code = ib.country_code
  AND rs.business_unit = ib.business_unit
  AND rs.item_id = ib.item_id
LEFT JOIN business_unit_baseline bu
  ON rs.country_code = bu.country_code
  AND rs.business_unit = bu.business_unit;