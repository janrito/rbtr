# Paraphrase report

LLM-generated concept queries: natural-language descriptions
of what each symbol does, without using identifier names.
These test vocabulary mismatch — the failure mode where a
developer searches with different words than the code uses.

## Summary

| metric          | value                    |
| --------------- | ------------------------ |
| model           | `openai:zai-org/GLM-5.1` |
| concept queries | 1157                     |

## Per repo

| slug               | n   |
| ------------------ | --- |
| anthropics__skills | 99  |
| astral-sh__uv      | 278 |
| badlogic__pi-mono  | 212 |
| django__django     | 359 |
| rbtr__rbtr         | 209 |

## Per language

| language   | n   |
| ---------- | --- |
| markdown   | 244 |
| python     | 193 |
| json       | 189 |
| javascript | 99  |
| typescript | 89  |
| rust       | 68  |
| yaml       | 66  |
| css        | 50  |
| rst        | 49  |
| html       | 45  |
| toml       | 34  |
| sql        | 31  |

## Examples

Randomly sampled symbols showing the source code (LLM input)
and the generated concept query (LLM output).

### `test_alter_text_field_to_date_field` (`django__django`)

````python
def test_alter_text_field_to_date_field(self):
        """
        #25002 - Test conversion of text field to date field.
        """
        with connection.schema_editor() as editor:
            editor.create_model(Note)
        Note.objects.create(info="1988-05-05")
        old_field = Note._meta.get_field("info")
        new_field = DateField(blank=True)
        new_field.set_attributes_from_name("info")
        with connection.schema_editor() as editor:
            editor.alter_field(Note, old_field, new_field, strict=True)
        # Make sure the field isn't nullable
        columns = self.column_classes(Note)
        self.assertFalse(columns["info"][1][6])
````

> **concept:** test converting a text field to a date field during database migration

### `const` (`astral-sh__uv`)

````json
"const": "aarch64-manylinux_2_37"
````

> **concept:** how to set the platform target for aarch64 manylinux builds

### `match_by_text` (`rbtr__rbtr`)

````python
def match_by_text(
        self,
        commit_sha: str,
        query: str,
        top_k: int = 10,
        *,
        repo_id: int,
        embedder: Embedder | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Semantic search: embed *query* then find similar chunks."""
        return scored_to_chunks(
            self._match_by_text(commit_sha, query, top_k, repo_id=repo_id, embedder=embedder)
        )
````

> **concept:** find semantically similar text chunks using a query embedder

### `django/forms/templates/django/forms/p.html` (`django__django`)

````html
{{ errors }}
{% if errors and not fields %}
  <p>{% for field in hidden_fields %}{{ field }}{% endfor %}</p>
{% endif %}
{% for field, errors in fields %}
  {{ errors }}
  <p{% with classes=field.css_classes %}{% if classes %} class="{{ classes }}"{% endif %}{% endwith %}>
    {% if field.label %}{{ field.label_tag }}{% endif %}
    {{ field }}
    {% if field.help_text %}
      <span class="helptext"{% if field.auto_id %} id="{{ field.auto_id }}_helptext"{% endif %}>{{ field.help_text|safe }}</span>
    {% endif %}
    {% if forloop.last %}
      {% for field in hidden_fields %}{{ field }}{% endfor %}
    {% endif %}
  </p>
{% endfor %}
{% if not fields and not errors %}
  {% for field in hidden_fields %}{{ field }}{% endfor %}
{% endif %}
````

> **concept:** how to render widget fields with error messages labels and help text
> inside paragraph tags

### `test_formset_with_ordering_and_deletion` (`django__django`)

````python
def test_formset_with_ordering_and_deletion(self):
        """FormSets with ordering + deletion."""
        ChoiceFormSet = formset_factory(Choice, can_order=True, can_delete=True)
        initial = [
            {"choice": "Calexico", "votes": 100},
            {"choice": "Fergie", "votes": 900},
            {"choice": "The Decemberists", "votes": 500},
        ]
        formset = ChoiceFormSet(initial=initial, auto_id=False, prefix="choices")
        self.assertHTMLEqual(
            "\n".join(form.as_ul() for form in formset.forms),
            '<li>Choice: <input type="text" name="choices-0-choice" value="Calexico">'
            "</li>"
            '<li>Votes: <input type="number" name="choices-0-votes" value="100"></li>'
            '<li>Order: <input type="number" name="choices-0-ORDER" value="1"></li>'
            '<li>Delete: <input type="checkbox" name="choices-0-DELETE"></li>'
            '<li>Choice: <input type="text" name="choices-1-choice" value="Fergie">'
            "</li>"
            '<li>Votes: <input type="number" name="choices-1-votes" value="900"></li>'
            '<li>Order: <input type="number" name="choices-1-ORDER" value="2"></li>'
            '<li>Delete: <input type="checkbox" name="choices-1-DELETE"></li>'
            '<li>Choice: <input type="text" name="choices-2-choice" '
            'value="The Decemberists"></li>'
            '<li>Votes: <input type="number" name="choices-2-votes" value="500"></li>'
            '<li>Order: <input type="number" name="choices-2-ORDER" value="3"></li>'
            '<li>Delete: <input type="checkbox" name="choices-2-DELETE"></li>'
            '<li>Choice: <input type="text" name="choices-3-choice"></li>'
            '<li>Votes: <input type="number" name="choices-3-votes"></li>'
            '<li>Order: <input type="number" name="choices-3-ORDER"></li>'
            '<li>Delete: <input type="checkbox" name="choices-3-DELETE"></li>',
        )
        # Let's delete Fergie, and put The Decemberists ahead of Calexico.
        data = {
            "choices-TOTAL_FORMS": "4",  # the number of forms rendered
            "choices-INITIAL_FORMS": "3",  # the number of forms with initial data
            "choices-MIN_NUM_FORMS": "0",  # min number of forms
            "choices-MAX_NUM_FORMS": "0",  # max number of forms
            "choices-0-choice": "Calexico",
            "choices-0-votes": "100",
            "choices-0-ORDER": "1",
            "choices-0-DELETE": "",
            "choices-1-choice": "Fergie",
            "choices-1-votes": "900",
            "choices-1-ORDER": "2",
            "choices-1-DELETE": "on",
            "choices-2-choice": "The Decemberists",
            "choices-2-votes": "500",
            "choices-2-ORDER": "0",
            "choices-2-DELETE": "",
            "choices-3-choice": "",
            "choices-3-votes": "",
            "choices-3-ORDER": "",
            "choices-3-DELETE": "",
        }
        formset = ChoiceFormSet(data, auto_id=False, prefix="choices")
        self.assertTrue(formset.is_valid())
        self.assertEqual(
            [form.cleaned_data for form in formset.ordered_forms],
            [
                {
                    "votes": 500,
                    "DELETE": False,
                    "ORDER": 0,
                    "choice": "The Decemberists",
                },
                {"votes": 100, "DELETE": False, "ORDER": 1, "choice": "Calexico"},
            ],
        )
        self.assertEqual(
            [form.cleaned_data for form in formset.deleted_forms],
            [{"votes": 900, "DELETE": True, "ORDER": 2, "choice": "Fergie"}],
        )
````

> **concept:** Django formset supporting both ordering and deletion of form entries

### `#toolbar form #searchbar` (`django__django`)

````css
#toolbar form #searchbar {
        flex: 1 0 auto;
        width: 0;
        height: 1.375rem;
        margin: 0 10px 0 6px;
    }
````

> **concept:** how to make a search bar expand to fill available space in a toolbar
> using flexbox

### `dist` (`astral-sh__uv`)

````toml
[profile.dist]
inherits = "release"
````

> **concept:** how to create a custom build configuration that inherits release settings

### `createDisabledPseudo` (`django__django`)

````javascript
/**
 * Returns a function to use in pseudos for :enabled/:disabled
 * @param {Boolean} disabled true for :disabled; false for :enabled
 */
function createDisabledPseudo( disabled ) {

	// Known :disabled false positives: fieldset[disabled] > legend:nth-of-type(n+2) :can-disable
	return function( elem ) {

		// Only certain elements can match :enabled or :disabled
		// https://html.spec.whatwg.org/multipage/scripting.html#selector-enabled
		// https://html.spec.whatwg.org/multipage/scripting.html#selector-disabled
		if ( "form" in elem ) {

			// Check for inherited disabledness on relevant non-disabled elements:
			// * listed form-associated elements in a disabled fieldset
			//   https://html.spec.whatwg.org/multipage/forms.html#category-listed
			//   https://html.spec.whatwg.org/multipage/forms.html#concept-fe-disabled
			// * option elements in a disabled optgroup
			//   https://html.spec.whatwg.org/multipage/forms.html#concept-option-disabled
			// All such elements have a "form" property.
			if ( elem.parentNode && elem.disabled === false ) {

				// Option elements defer to a parent optgroup if present
				if ( "label" in elem ) {
					if ( "label" in elem.parentNode ) {
						return elem.parentNode.disabled === disabled;
					} else {
						return elem.disabled === disabled;
					}
				}

				// Support: IE 6 - 11+
				// Use the isDisabled shortcut property to check for disabled fieldset ancestors
				return elem.isDisabled === disabled ||

					// Where there is no isDisabled, check manually
					elem.isDisabled !== !disabled &&
						inDisabledFieldset( elem ) === disabled;
			}

			return elem.disabled === disabled;

		// Try to winnow out elements that can't be disabled before trusting the disabled property.
		// Some victims get caught in our net (label, legend, menu, track), but it shouldn't
		// even exist on them, let alone have a boolean value.
		} else if ( "label" in elem ) {
			return elem.disabled === disabled;
		}

		// Remaining elements are neither :enabled nor :disabled
		return false;
	};
}
````

> **concept:** check if a form element is enabled or disabled for CSS pseudo-selector
> matching

### `partial_cmp` (`astral-sh__uv`)

````rust
fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
````

> **concept:** Compare two values for partial ordering

### `mrr` (`rbtr__rbtr`)

````json
"mrr": 0.4622014177839421
````

> **concept:** how to evaluate search ranking quality using reciprocal rank scores
