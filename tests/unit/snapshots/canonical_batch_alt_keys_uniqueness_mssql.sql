SELECT count(*) AS total_rows, count(DISTINCT [#mp_probe].key_hash) AS distinct_keys 
FROM [#mp_probe]
