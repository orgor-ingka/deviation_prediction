### Author: Niclas Lavesson
  -- Using table from Divya Mondal
# Date: 2025-06-21
# This table is used in the deviation predictions BQ script

CREATE OR REPLACE TABLE `ingka-ff-somdata-prod.OMDA_Analytics.item_id_service_levels` AS
WITH tab1 AS (
  SELECT DISTINCT CONCAT(REGEXP_EXTRACT(BU, r'-(\w+)'), '.', REGEXP_EXTRACT(BU, r'^(\d+)')) AS BU, article_id, sl_change_journey
  FROM `ingka-ff-somdata-prod.OMDA_Analytics.Availability_Art_SLChngJourney`
  WHERE sl_change_flag = 'Y'
),

tab2 AS(
SELECT
  BU AS ship_node,
  article_id AS item_id,
  REGEXP_EXTRACT(change, r'(\d+)→\d+ on \d{4}-\d{2}-\d{2}') AS from_SL,
  REGEXP_EXTRACT(change, r'\d+→(\d+) on \d{4}-\d{2}-\d{2}') AS to_SL,
  CAST(REGEXP_EXTRACT(change, r'\d+→\d+ on (\d{4}-\d{2}-\d{2})') AS DATE) AS change_date
FROM (
  SELECT
    BU,
    article_id,
    TRIM(change) AS change
  FROM
    tab1,
    UNNEST(SPLIT(sl_change_journey, ',')) AS change)),

tab3 AS (
  SELECT DISTINCT CONCAT(REGEXP_EXTRACT(BU, r'-(\w+)'), '.', REGEXP_EXTRACT(BU, r'^(\d+)')) AS ship_node
        ,article_id AS item_id
        ,SUBSTR(no_change_journey,1,1) AS from_SL
        ,SUBSTR(no_change_journey,3,1) AS to_SL
        ,CAST('2024-07-01' AS DATE) AS change_date
  FROM `ingka-ff-somdata-prod.OMDA_Analytics.Availability_Art_SLChngJourney`
  WHERE sl_change_flag = 'N')

SELECT * FROM tab2
UNION ALL
SELECT * FROM tab3;


