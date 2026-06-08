# Paraphrase report

LLM-generated concept queries: natural-language descriptions
of what each symbol does, without using identifier names.
These test vocabulary mismatch — the failure mode where a
developer searches with different words than the code uses.

## Summary

| metric          | value                    |
| --------------- | ------------------------ |
| model           | `openai:zai-org/GLM-5.1` |
| concept queries | 1122                     |

## Per repo

| slug               | n   |
| ------------------ | --- |
| anthropics__skills | 99  |
| astral-sh__uv      | 276 |
| badlogic__pi-mono  | 213 |
| django__django     | 357 |
| rbtr__rbtr         | 177 |

## Per language

| language   | n   |
| ---------- | --- |
| markdown   | 240 |
| python     | 193 |
| json       | 192 |
| javascript | 96  |
| typescript | 88  |
| rust       | 69  |
| yaml       | 67  |
| css        | 50  |
| rst        | 48  |
| html       | 45  |
| toml       | 34  |

## Examples

Randomly sampled symbols showing the source code (LLM input)
and the generated concept query (LLM output).

### `count` (`django__django`)

````json
"count": 3
````

> **concept:** how to configure the number of database records to create

### `items` (`astral-sh__uv`)

````json
"items": {
        "$ref": "#/definitions/StaticMetadata"
      }
````

> **concept:** how to reference a reusable schema definition for array elements

### `language` (`rbtr__rbtr`)

````json
"language": "python"
````

> **concept:** how to set which programming environment or coding syntax a project uses
> in config

### `formfield` (`django__django`)

````python
def formfield(self, **kwargs):
        kwargs.setdefault("form_class", self.form_field)
        return super().formfield(**kwargs)
````

> **concept:** set default form class when creating a form widget from a model attribute

### `tests/auth_tests/templates/context_processors/auth_attrs_test_access.html` (`django__django`)

````html
{% if session_accessed %}Session accessed{% else %}Session not accessed{% endif %}
````

> **concept:** how to check if Django session has been accessed in a template

### `.select2-container--admin-autocomplete.select2-container--disabled …` (`django__django`)

**symbol:**

```text
.select2-container--admin-autocomplete.select2-container--disabled
    .select2-selection__choice__remove
```

````css
.select2-container--admin-autocomplete.select2-container--disabled
    .select2-selection__choice__remove {
    display: none;
}
````

> **concept:** how to hide the remove button on disabled select2 choices

### `Preview features` (`astral-sh__uv`)

````markdown
### Preview features

- Add module owners to `uv workspace metadata`
  ([#19122](https://github.com/astral-sh/uv/pull/19122))
- Do not allow `uv venv --clear` to remove non-virtual environments
  ([#19595](https://github.com/astral-sh/uv/pull/19595))
````

> **concept:** how to prevent uv venv clear from deleting non-virtual environments

### `form .aligned p.help:last-child, …` (`django__django`)

**symbol:**

```text
form .aligned p.help:last-child,
form .aligned div.help:last-child
```

````css
form .aligned p.help:last-child,
form .aligned div.help:last-child {
    margin-bottom: 0;
    padding-bottom: 0;
}
````

> **concept:** how to remove bottom margin from last help text in an aligned layout

### `trailing_zero_equality` (`astral-sh__uv`)

````rust
fn trailing_zero_equality() {
        let equal = [
            // Non-`~=` operators: trailing zeros are insignificant.
            (">=3.3", ">=3.3.0"),
            ("<2", "<2.0.0"),
            ("==1.2", "==1.2.0"),
            // Identical `~=` specifiers.
            ("~=2.2.0", "~=2.2.0"),
        ];
        for (a, b) in equal {
            let a = VersionSpecifier::from_str(a).unwrap();
            let b = VersionSpecifier::from_str(b).unwrap();
            assert_eq!(a, b);
        }

        let not_equal = [
            // PEP 440 forward-compat examples.
            ("~=2.2", "~=2.2.0"),
            ("~=1.4.5", "~=1.4.5.0"),
            // Same release, different suffix.
            ("~=2.2.post3", "~=2.2.post5"),
            // Different release length with matching suffix.
            ("~=2.2.post3", "~=2.2.0.post3"),
        ];
        for (a, b) in not_equal {
            let a = VersionSpecifier::from_str(a).unwrap();
            let b = VersionSpecifier::from_str(b).unwrap();
            assert_ne!(a, b);
        }
    }
````

> **concept:** test that trailing zeros in PEP 440 version specifiers are treated
> correctly for equality comparison

### `slug` (`rbtr__rbtr`)

````json
"slug": "anthropics__skills"
````

> **concept:** how to set the unique identifier for a skills configuration
