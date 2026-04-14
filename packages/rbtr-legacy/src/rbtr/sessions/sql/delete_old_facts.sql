DELETE FROM facts
WHERE last_confirmed_at < ?;
