SELECT block_size * total_blocks AS bytes
FROM pragma_database_size()
