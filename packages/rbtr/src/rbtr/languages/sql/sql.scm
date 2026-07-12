; DDL/ALTER/DROP verbs, grouped by chunk kind and name location.
;
; object_reference names are left unanchored: each verb has exactly one
; direct object_reference, and leaving it unanchored keeps IF NOT EXISTS /
; OR REPLACE / TEMP working (anchoring to a keyword breaks when those
; intervene). A bare (identifier) name (index/schema/role/…) is used when
; the object_reference, if any, is a secondary reference (e.g. an index's
; ON table). Verified against tree-sitter-sql 0.3.11.

[
  (create_table (object_reference name: (identifier) @_cls_name))
  (create_view (object_reference name: (identifier) @_cls_name))
  (create_materialized_view (object_reference name: (identifier) @_cls_name))
  (create_type (object_reference name: (identifier) @_cls_name))
] @class

(create_sequence (object_reference name: (identifier) @_var_name)) @variable

[
  (create_function (object_reference name: (identifier) @_fn_name))
  (alter_table (object_reference name: (identifier) @_fn_name))
  (alter_view (object_reference name: (identifier) @_fn_name))
  (alter_sequence (object_reference name: (identifier) @_fn_name))
  (drop_table (object_reference name: (identifier) @_fn_name))
  (drop_view (object_reference name: (identifier) @_fn_name))
  (drop_sequence (object_reference name: (identifier) @_fn_name))
  (drop_function (object_reference name: (identifier) @_fn_name))
  (drop_type (object_reference name: (identifier) @_fn_name))
] @function

[
  (create_index (identifier) @_var_name)
  (create_schema (identifier) @_var_name)
  (create_database (identifier) @_var_name)
  (create_extension (identifier) @_var_name)
  (create_role (identifier) @_var_name)
] @variable

[
  (alter_index (identifier) @_fn_name)
  (alter_schema (identifier) @_fn_name)
  (alter_database (identifier) @_fn_name)
  (alter_role (identifier) @_fn_name)
  (alter_type (identifier) @_fn_name)
  (drop_index (identifier) @_fn_name)
  (drop_schema (identifier) @_fn_name)
  (drop_database (identifier) @_fn_name)
  (drop_role (identifier) @_fn_name)
  (drop_extension (identifier) @_fn_name)
] @function

; DML statements, CTEs, triggers, and set operations — structures the DDL
; groups can't express. DML patterns are anchored at (program (statement …))
; so each top-level statement yields exactly one chunk: select and delete
; keep their target in a sibling from, so the whole statement is captured.
; A set_operation (UNION/INTERSECT/EXCEPT) has no single table → <anonymous>.

(create_trigger (keyword_trigger) . (object_reference name: (identifier) @_var_name)) @variable

(cte (identifier) @_fn_name) @function

(program (statement (select)
  (from (relation (object_reference name: (identifier) @_fn_name)))?) @function)
(program (statement (set_operation) @function))
(program (statement (insert (object_reference name: (identifier) @_fn_name))) @function)
(program (statement (update (relation (object_reference name: (identifier) @_fn_name)))) @function)
(program (statement (delete) (from (object_reference name: (identifier) @_fn_name))) @function)
