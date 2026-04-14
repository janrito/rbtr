-- Cross-session aggregate overhead costs.
-- No params.
SELECT
  SUM(
    CASE
      WHEN fragment_kind = 'overhead-compaction'
        THEN COALESCE(input_tokens, 0)
      ELSE 0
    END
  ) AS compaction_input_tokens,
  SUM(
    CASE
      WHEN fragment_kind = 'overhead-compaction'
        THEN COALESCE(output_tokens, 0)
      ELSE 0
    END
  ) AS compaction_output_tokens,
  SUM(
    CASE
      WHEN fragment_kind = 'overhead-compaction'
        THEN COALESCE(cost, 0)
      ELSE 0
    END
  ) AS compaction_cost,
  COUNT(
    CASE
      WHEN fragment_kind = 'overhead-compaction' THEN 1
    END
  ) AS compaction_count,
  SUM(
    CASE
      WHEN fragment_kind = 'overhead-fact-extraction'
        THEN COALESCE(input_tokens, 0)
      ELSE 0
    END
  ) AS fact_extraction_input_tokens,
  SUM(
    CASE
      WHEN fragment_kind = 'overhead-fact-extraction'
        THEN COALESCE(output_tokens, 0)
      ELSE 0
    END
  ) AS fact_extraction_output_tokens,
  SUM(
    CASE
      WHEN fragment_kind = 'overhead-fact-extraction'
        THEN COALESCE(cost, 0)
      ELSE 0
    END
  ) AS fact_extraction_cost,
  COUNT(
    CASE
      WHEN fragment_kind = 'overhead-fact-extraction' THEN 1
    END
  ) AS fact_extraction_count
FROM fragments
WHERE
  fragment_kind IN ('overhead-compaction', 'overhead-fact-extraction')
  AND status = 'complete'
