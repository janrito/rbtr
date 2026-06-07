-- Check whether a repo has any completed builds.
SELECT 1
WHERE
  EXISTS (
    SELECT 1
    FROM indexed_commits
    WHERE repo_id = $repo_id
  )
