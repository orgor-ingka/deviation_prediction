### Author: Chris Brewer and Niclas Lavesson
# This creates a calendar for the unit operation hours.

CREATE OR REPLACE TABLE `ingka-ff-somdata-prod.OMDA_Analytics.unit_operation_hours_2025` AS
WITH tab1 AS
(SELECT b.organization_code
      ,b.calendar_id
      ,b.calendar_description
      ,a.*
FROM `ingka-ff-somdata-prod.common_dim_no_pii.vw_calendar_shift_eff_perd_dim` AS a
LEFT JOIN `ingka-ff-somdata-prod.common_dim_no_pii.vw_calendar_dim` AS b
ON a.calendar_key = b.calendar_key),

tab2 AS
(SELECT a.*
        ,b.shift_name
        ,b.shift_start_time
        ,b.shift_end_time
        ,b.monday_valid
        ,b.tuesday_valid
        ,b.wednesday_valid
        ,b.thursday_valid
        ,b.friday_valid
        ,b.saturday_valid
        ,b.sunday_valid
FROM tab1 AS a
LEFT JOIN `ingka-ff-somdata-prod.common_dim_no_pii.vw_calendar_shift_dim` AS b
ON a.calendar_shift_eff_perd_key = b.calendar_shift_eff_perd_key),
 
tab3 AS
(SELECT organization_code
      ,calendar_description
      ,createts
      ,modifyts
      ,lockid
      ,LEAD(modifyts) OVER (PARTITION BY organization_code, shift_name ORDER BY modifyts ASC) AS lead_modifyts
      ,shift_name
      ,monday_valid
      ,tuesday_valid
      ,wednesday_valid
      ,thursday_valid
      ,friday_valid
      ,saturday_valid
      ,sunday_valid
FROM tab2
WHERE organization_code IS NOT NULL
AND shift_name IS NOT NULL
AND shift_name LIKE ('%:%-%:%')
AND calendar_id = 'REC'
QUALIFY ROW_NUMBER() OVER (PARTITION BY organization_code, shift_name, modifyts) = 1
),
 
tab4 AS
(SELECT organization_code
      ,shift_name
      ,modifyts AS shifts_valid_from
      ,CASE
            WHEN lead_modifyts IS NULL THEN TIMESTAMP('2025-12-31')
            ELSE lead_modifyts
      END AS shifts_valid_to
      ,CASE
            WHEN monday_valid = 'Y' THEN shift_name
            ELSE NULL
      END AS monday_shift
      ,CASE
            WHEN tuesday_valid = 'Y' THEN shift_name
            ELSE NULL
      END AS tuesday_shift
      ,CASE
            WHEN wednesday_valid = 'Y' THEN shift_name
            ELSE NULL
      END AS wednesday_shift
      ,CASE
            WHEN thursday_valid = 'Y' THEN shift_name
            ELSE NULL
      END AS thursday_shift
      ,CASE
            WHEN friday_valid = 'Y' THEN shift_name
            ELSE NULL
      END AS friday_shift
      ,CASE
            WHEN saturday_valid = 'Y' THEN shift_name
            ELSE NULL
      END AS saturday_shift
      ,CASE
            WHEN sunday_valid = 'Y' THEN shift_name
            ELSE NULL
      END AS sunday_shift
FROM tab3
ORDER BY organization_code, modifyts ASC),

tab4a AS (
      SELECT organization_code
               ,shifts_valid_from
               ,shifts_valid_to
            ,MAX(monday_shift) AS monday_shift
            ,MAX(tuesday_shift) AS tuesday_shift
            ,MAX(wednesday_shift) AS wednesday_shift
            ,MAX(thursday_shift) AS thursday_shift
            ,MAX(friday_shift) AS friday_shift
            ,MAX(saturday_shift) AS saturday_shift
            ,MAX(sunday_shift) AS sunday_shift
      FROM tab4
      GROUP BY organization_code, shifts_valid_from, shifts_valid_to
),

calendar AS (
  -- Generate dates from Sept 1, 2024 to December 31st, 2025
  SELECT date AS calendar_date
  FROM UNNEST(GENERATE_DATE_ARRAY(DATE '2024-09-01', DATE '2025-12-31')) AS date
),
 
calendar2 AS (
SELECT calendar_date, UPPER(FORMAT_DATE('%A', calendar_date)) AS day_name
FROM calendar),

unpivoted_shifts AS (
  -- Flatten day-specific shift fields into day-of-week + shift string
  SELECT organization_code, shifts_valid_from, shifts_valid_to, 'MONDAY' AS day_of_week, monday_shift AS shift FROM tab4a
  UNION ALL SELECT organization_code, shifts_valid_from, shifts_valid_to, 'TUESDAY', tuesday_shift FROM tab4a
  UNION ALL SELECT organization_code, shifts_valid_from, shifts_valid_to, 'WEDNESDAY', wednesday_shift FROM tab4a
  UNION ALL SELECT organization_code, shifts_valid_from, shifts_valid_to, 'THURSDAY', thursday_shift FROM tab4a
  UNION ALL SELECT organization_code, shifts_valid_from, shifts_valid_to, 'FRIDAY', friday_shift FROM tab4a
  UNION ALL SELECT organization_code, shifts_valid_from, shifts_valid_to, 'SATURDAY', saturday_shift FROM tab4a
  UNION ALL SELECT organization_code, shifts_valid_from, shifts_valid_to, 'SUNDAY', sunday_shift FROM tab4a
),

unpivoted_shifts2 AS (
SELECT * EXCEPT(shift)
      ,CASE
        WHEN shift IS NULL THEN '00:00:00-00:00:00' ELSE shift
      END AS shift  
FROM unpivoted_shifts),

date_expanded AS (
SELECT 
  s.organization_code,
  c.calendar_date,
  s.day_of_week,
  s.shift,
  s.shifts_valid_from,
  s.shifts_valid_to
FROM unpivoted_shifts2 s
INNER JOIN calendar2 c
  ON TIMESTAMP(c.calendar_date) BETWEEN TIMESTAMP(s.shifts_valid_from) AND TIMESTAMP(s.shifts_valid_to)
  AND UPPER(c.day_name) = UPPER(s.day_of_week)),

date_expanded_cleaned AS
(SELECT calendar_date
      ,organization_code
      ,CASE
            WHEN LENGTH(shift) = 11 THEN CONCAT(SUBSTR(shift,0,5),':00-',SUBSTR(shift,7),':00')
            ELSE shift
      END AS shift
FROM date_expanded),
 
parsed_shifts AS (
  -- Parse the shift string into start and end time
  SELECT
    organization_code,
    calendar_date,
    shift,
    -- Use SAFE.PARSE_TIME in case of NULLs (e.g. closed days)
    SAFE.PARSE_TIME('%H:%M:%S', SPLIT(shift, '-')[OFFSET(0)]) AS shift_start,
    SAFE.PARSE_TIME('%H:%M:%S', SPLIT(shift, '-')[OFFSET(1)]) AS shift_end
  FROM date_expanded_cleaned
),

parsed_shifts2 AS (
SELECT organization_code
      ,calendar_date
      ,MIN(shift_start) AS shift_start
      ,MAX(shift_end) AS shift_end
FROM parsed_shifts
GROUP BY organization_code
      ,calendar_date
      ,shift
)
 
-- Final output
SELECT * 
FROM parsed_shifts2
ORDER BY organization_code, calendar_date