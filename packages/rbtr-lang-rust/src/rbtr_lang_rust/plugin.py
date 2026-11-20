"""Rust language plugin.

Provides full support: functions, structs, enums, unions, type
aliases, traits, modules, impl blocks, `macro_rules!` definitions,
and `use` declaration extraction. Traits, modules, and impl blocks
form naming scopes; trait and impl members are methods.

Extracted chunks::

    fn hello() {}                   → function "hello", scope ""
    struct User { name: String }    → class "User", scope ""
    enum Color { Red, Green }       → class "Color", scope ""
    impl Svc {
        fn start(&self) {}          → method "start", scope "Svc"
    }

    use std::collections::HashMap
        → import, metadata {module: "std::collections", names: "HashMap"}
    use crate::models::{Chunk, Edge}
        → import, metadata {module: "crate::models", names: "Chunk,Edge"}
    use super::utils
        → import, metadata {module: "../", names: "utils"}
"""

from __future__ import annotations

from rbtr.domain.edge_frames import ModuleStyle
from rbtr.languages.registration import LanguageRegistration, QueryExtraction, load_query

rust = LanguageRegistration(
    id="rust",
    extensions=frozenset({".rs"}),
    grammar_module="tree_sitter_rust",
    extraction=QueryExtraction(
        query=load_query(__package__, "rust") + load_query(__package__, "references"),
        scope_types=frozenset({"impl_item", "struct_item", "trait_item", "mod_item", "enum_item"}),
        class_scope_types=frozenset({"impl_item", "struct_item", "trait_item"}),
    ),
    index_files=frozenset({"mod.rs"}),
    module_style=ModuleStyle.SCOPED,
    path_substitutions=(("crate::", "src/"),),
    parent_module="super",
    current_module="self",
    directory_module_files=frozenset({"mod.rs", "lib.rs", "main.rs"}),
    extraction_serial=12,
)
