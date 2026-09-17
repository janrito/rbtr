# Paraphrase report

LLM-generated concept queries: natural-language descriptions
of what each symbol does, without using identifier names.
These test vocabulary mismatch — the failure mode where a
developer searches with different words than the code uses.

## Summary

| metric          | value                                         |
| --------------- | --------------------------------------------- |
| model           | `openai-chat:deepseek/deepseek-v4-flash-0731` |
| concept queries | 1576                                          |

## Per repo

| slug               | n   |
| ------------------ | --- |
| anthropics__skills | 307 |
| astral-sh__uv      | 411 |
| badlogic__pi-mono  | 278 |
| django__django     | 336 |
| rbtr__rbtr         | 244 |

## Per language

| language   | n   |
| ---------- | --- |
| python     | 370 |
| typescript | 206 |
| javascript | 178 |
| bash       | 171 |
| css        | 145 |
| rust       | 122 |
| json       | 99  |
| markdown   | 99  |
| yaml       | 41  |
| rst        | 35  |
| toml       | 33  |
| sql        | 31  |
| plaintext  | 30  |
| html       | 16  |

## Examples

Randomly sampled symbols showing the source code (LLM input)
and the generated concept query (LLM output).

### `LAST_MODIFIED` (`django__django`)

````python
LAST_MODIFIED = datetime(2007, 10, 21, 23, 21, 47)
````

> **concept:** When was this module last updated or changed

### `` (`astral-sh__uv`)

````rst
---
source: crates/uv-requirements-txt/src/lib.rs
expression: actual
---
RequirementsTxt {
    requirements: [
        RequirementEntry {
            requirement: Named(
                Requirement {
                    name: PackageName(
                        "tomli",
                    ),
                    extras: [],
                    version_or_url: None,
                    marker: true,
                    origin: Some(
                        File(
                            "<REQUIREMENTS_DIR>/include-b.txt",
                        ),
                    ),
                },
            ),
            hashes: [],
        },
        RequirementEntry {
            requirement: Named(
                Requirement {
                    name: PackageName(
                        "numpy",
                    ),
                    extras: [],
                    version_or_url: Some(
                        VersionSpecifier(
                            VersionSpecifiers(
                                [
                                    VersionSpecifier {
                                        operator: Equal,
                                        version: "1.24.2",
                                    },
                                ],
                            ),
                        ),
                    ),
                    marker: true,
                    origin: Some(
                        File(
                            "<REQUIREMENTS_DIR>/include-a.txt",
                        ),
                    ),
                },
            ),
            hashes: [],
        },
    ],
    constraints: [],
    editables: [],
    index_url: None,
    extra_index_urls: [],
    find_links: [],
    no_index: false,
    no_binary: None,
    only_binary: None,
}
````

> **concept:** how to include requirements from another file

### `test_roundtrip` (`rbtr__rbtr`)

````python
def test_roundtrip(scenario: MessageScenario) -> None:
    """Model survives serialise → deserialise via its adapter."""
    model = scenario.adapter.validate_json(scenario.raw)
    roundtripped = scenario.adapter.validate_json(model.model_dump_json())
    assert type(roundtripped) is type(model)
````

> **concept:** verify a serialized model deserializes back to the same type

### `setDocument` (`django__django`)

````javascript
/**
 * Sets document-related variables once based on the current document
 * @param {Element|Object} [node] An element or document object to use to set the document
 * @returns {Object} Returns the current document
 */
function setDocument( node ) {
	var subWindow,
		doc = node ? node.ownerDocument || node : preferredDoc;

	// Return early if doc is invalid or already selected
	// Support: IE 11+, Edge 17 - 18+
	// IE/Edge sometimes throw a "Permission denied" error when strict-comparing
	// two documents; shallow comparisons work.
	// eslint-disable-next-line eqeqeq
	if ( doc == document || doc.nodeType !== 9 || !doc.documentElement ) {
		return document;
	}

	// Update global variables
	document = doc;
	documentElement = document.documentElement;
	documentIsHTML = !jQuery.isXMLDoc( document );

	// Support: iOS 7 only, IE 9 - 11+
	// Older browsers didn't support unprefixed `matches`.
	matches = documentElement.matches ||
		documentElement.webkitMatchesSelector ||
		documentElement.msMatchesSelector;

	// Support: IE 9 - 11+, Edge 12 - 18+
	// Accessing iframe documents after unload throws "permission denied" errors
	// (see trac-13936).
	// Limit the fix to IE & Edge Legacy; despite Edge 15+ implementing `matches`,
	// all IE 9+ and Edge Legacy versions implement `msMatchesSelector` as well.
	if ( documentElement.msMatchesSelector &&

		// Support: IE 11+, Edge 17 - 18+
		// IE/Edge sometimes throw a "Permission denied" error when strict-comparing
		// two documents; shallow comparisons work.
		// eslint-disable-next-line eqeqeq
		preferredDoc != document &&
		( subWindow = document.defaultView ) && subWindow.top !== subWindow ) {

		// Support: IE 9 - 11+, Edge 12 - 18+
		subWindow.addEventListener( "unload", unloadHandler );
	}

	// Support: IE <10
	// Check if getElementById returns elements by name
	// The broken getElementById methods don't pick up programmatically-set names,
	// so use a roundabout getElementsByName test
	support.getById = assert( function( el ) {
		documentElement.appendChild( el ).id = jQuery.expando;
		return !document.getElementsByName ||
			!document.getElementsByName( jQuery.expando ).length;
	} );

	// Support: IE 9 only
	// Check to see if it's possible to do matchesSelector
	// on a disconnected node.
	support.disconnectedMatch = assert( function( el ) {
		return matches.call( el, "*" );
	} );

	// Support: IE 9 - 11+, Edge 12 - 18+
	// IE/Edge don't support the :scope pseudo-class.
	support.scope = assert( function() {
		return document.querySelectorAll( ":scope" );
	} );

	// Support: Chrome 105 - 111 only, Safari 15.4 - 16.3 only
	// Make sure the `:has()` argument is parsed unforgivingly.
	// We include `*` in the test to detect buggy implementations that are
	// _selectively_ forgiving (specifically when the list includes at least
	// one valid selector).
	// Note that we treat complete lack of support for `:has()` as if it were
	// spec-compliant support, which is fine because use of `:has()` in such
	// environments will fail in the qSA path and fall back to jQuery traversal
	// anyway.
	support.cssHas = assert( function() {
		try {
			document.querySelector( ":has(*,:jqfake)" );
			return false;
		} catch ( e ) {
			return true;
		}
	} );

	// ID filter and find
	if ( support.getById ) {
		Expr.filter.ID = function( id ) {
			var attrId = id.replace( runescape, funescape );
			return function( elem ) {
				return elem.getAttribute( "id" ) === attrId;
			};
		};
		Expr.find.ID = function( id, context ) {
			if ( typeof context.getElementById !== "undefined" && documentIsHTML ) {
				var elem = context.getElementById( id );
				return elem ? [ elem ] : [];
			}
		};
	} else {
		Expr.filter.ID =  function( id ) {
			var attrId = id.replace( runescape, funescape );
			return function( elem ) {
				var node = typeof elem.getAttributeNode !== "undefined" &&
					elem.getAttributeNode( "id" );
				return node && node.value === attrId;
			};
		};

		// Support: IE 6 - 7 only
		// getElementById is not reliable as a find shortcut
		Expr.find.ID = function( id, context ) {
			if ( typeof context.getElementById !== "undefined" && documentIsHTML ) {
				var node, i, elems,
					elem = context.getElementById( id );

				if ( elem ) {

					// Verify the id attribute
					node = elem.getAttributeNode( "id" );
					if ( node && node.value === id ) {
						return [ elem ];
					}

					// Fall back on getElementsByName
					elems = context.getElementsByName( id );
					i = 0;
					while ( ( elem = elems[ i++ ] ) ) {
						node = elem.getAttributeNode( "id" );
						if ( node && node.value === id ) {
							return [ elem ];
						}
					}
				}

				return [];
			}
		};
	}

	// Tag
	Expr.find.TAG = function( tag, context ) {
		if ( typeof context.getElementsByTagName !== "undefined" ) {
			return context.getElementsByTagName( tag );

		// DocumentFragment nodes don't have gEBTN
		} else {
			return context.querySelectorAll( tag );
		}
	};

	// Class
	Expr.find.CLASS = function( className, context ) {
		if ( typeof context.getElementsByClassName !== "undefined" && documentIsHTML ) {
			return context.getElementsByClassName( className );
		}
	};

	/* QSA/matchesSelector
	---------------------------------------------------------------------- */

	// QSA and matchesSelector support

	rbuggyQSA = [];

	// Build QSA regex
	// Regex strategy adopted from Diego Perini
	assert( function( el ) {

		var input;

		documentElement.appendChild( el ).innerHTML =
			"<a id='" + expando + "' href='' disabled='disabled'></a>" +
			"<select id='" + expando + "-\r\\' disabled='disabled'>" +
			"<option selected=''></option></select>";

		// Support: iOS <=7 - 8 only
		// Boolean attributes and "value" are not treated correctly in some XML documents
		if ( !el.querySelectorAll( "[selected]" ).length ) {
			rbuggyQSA.push( "\\[" + whitespace + "*(?:value|" + booleans + ")" );
		}

		// Support: iOS <=7 - 8 only
		if ( !el.querySelectorAll( "[id~=" + expando + "-]" ).length ) {
			rbuggyQSA.push( "~=" );
		}

		// Support: iOS 8 only
		// https://bugs.webkit.org/show_bug.cgi?id=136851
		// In-page `selector#id sibling-combinator selector` fails
		if ( !el.querySelectorAll( "a#" + expando + "+*" ).length ) {
			rbuggyQSA.push( ".#.+[+~]" );
		}

		// Support: Chrome <=105+, Firefox <=104+, Safari <=15.4+
		// In some of the document kinds, these selectors wouldn't work natively.
		// This is probably OK but for backwards compatibility we want to maintain
		// handling them through jQuery traversal in jQuery 3.x.
		if ( !el.querySelectorAll( ":checked" ).length ) {
			rbuggyQSA.push( ":checked" );
		}

		// Support: Windows 8 Native Apps
		// The type and name attributes are restricted during .innerHTML assignment
		input = document.createElement( "input" );
		input.setAttribute( "type", "hidden" );
		el.appendChild( input ).setAttribute( "name", "D" );

		// Support: IE 9 - 11+
		// IE's :disabled selector does not pick up the children of disabled fieldsets
		// Support: Chrome <=105+, Firefox <=104+, Safari <=15.4+
		// In some of the document kinds, these selectors wouldn't work natively.
		// This is probably OK but for backwards compatibility we want to maintain
		// handling them through jQuery traversal in jQuery 3.x.
		documentElement.appendChild( el ).disabled = true;
		if ( el.querySelectorAll( ":disabled" ).length !== 2 ) {
			rbuggyQSA.push( ":enabled", ":disabled" );
		}

		// Support: IE 11+, Edge 15 - 18+
		// IE 11/Edge don't find elements on a `[name='']` query in some cases.
		// Adding a temporary attribute to the document before the selection works
		// around the issue.
		// Interestingly, IE 10 & older don't seem to have the issue.
		input = document.createElement( "input" );
		input.setAttribute( "name", "" );
		el.appendChild( input );
		if ( !el.querySelectorAll( "[name='']" ).length ) {
			rbuggyQSA.push( "\\[" + whitespace + "*name" + whitespace + "*=" +
				whitespace + "*(?:''|\"\")" );
		}
	} );

	if ( !support.cssHas ) {

		// Support: Chrome 105 - 110+, Safari 15.4 - 16.3+
		// Our regular `try-catch` mechanism fails to detect natively-unsupported
		// pseudo-classes inside `:has()` (such as `:has(:contains("Foo"))`)
		// in browsers that parse the `:has()` argument as a forgiving selector list.
		// https://drafts.csswg.org/selectors/#relational now requires the argument
		// to be parsed unforgivingly, but browsers have not yet fully adjusted.
		rbuggyQSA.push( ":has" );
	}

	rbuggyQSA = rbuggyQSA.length && new RegExp( rbuggyQSA.join( "|" ) );

	/* Sorting
	---------------------------------------------------------------------- */

	// Document order sorting
	sortOrder = function( a, b ) {

		// Flag for duplicate removal
		if ( a === b ) {
			hasDuplicate = true;
			return 0;
		}

		// Sort on method existence if only one input has compareDocumentPosition
		var compare = !a.compareDocumentPosition - !b.compareDocumentPosition;
		if ( compare ) {
			return compare;
		}

		// Calculate position if both inputs belong to the same document
		// Support: IE 11+, Edge 17 - 18+
		// IE/Edge sometimes throw a "Permission denied" error when strict-comparing
		// two documents; shallow comparisons work.
		// eslint-disable-next-line eqeqeq
		compare = ( a.ownerDocument || a ) == ( b.ownerDocument || b ) ?
			a.compareDocumentPosition( b ) :

			// Otherwise we know they are disconnected
			1;

		// Disconnected nodes
		if ( compare & 1 ||
			( !support.sortDetached && b.compareDocumentPosition( a ) === compare ) ) {

			// Choose the first element that is related to our preferred document
			// Support: IE 11+, Edge 17 - 18+
			// IE/Edge sometimes throw a "Permission denied" error when strict-comparing
			// two documents; shallow comparisons work.
			// eslint-disable-next-line eqeqeq
			if ( a === document || a.ownerDocument == preferredDoc &&
				find.contains( preferredDoc, a ) ) {
				return -1;
			}

			// Support: IE 11+, Edge 17 - 18+
			// IE/Edge sometimes throw a "Permission denied" error when strict-comparing
			// two documents; shallow comparisons work.
			// eslint-disable-next-line eqeqeq
			if ( b === document || b.ownerDocument == preferredDoc &&
				find.contains( preferredDoc, b ) ) {
				return 1;
			}

			// Maintain original order
			return sortInput ?
				( indexOf.call( sortInput, a ) - indexOf.call( sortInput, b ) ) :
				0;
		}

		return compare & 4 ? -1 : 1;
	};

	return document;
}
````

> **concept:** setup and cache global document references and selector support flags

### `` (`django__django`)

````bash
if [[ -n "${GPG_KEY:-}" ]]; then
    gpg --recv-keys "${GPG_KEY}"
fi
````

> **concept:** how to import a GPG key into the keyring when building

### `wss` (`badlogic__pi-mono`)

````javascript
wss = new WebSocketServer({ server, clientTracking: true })
````

> **concept:** how to enable tracking of connected websocket clients

### `MCPConnectionHTTP` (`anthropics__skills`)

````python
class MCPConnectionHTTP(MCPConnection):
    """MCP connection using Streamable HTTP."""

    def __init__(self, url: str, headers: dict[str, str] = None):
        super().__init__()
        self.url = url
        self.headers = headers or {}

    def _create_context(self):
        return streamablehttp_client(url=self.url, headers=self.headers)
````

> **concept:** establish an MCP connection over streamable HTTP

### `cleanup` (`badlogic__pi-mono`)

````bash
# Keep script running
cleanup() {
  echo "Shutting down..."
  kill $NODE_PID 2>/dev/null || true
  kill $TUNNEL_PID 2>/dev/null || true
  exit 0
}
````

> **concept:** gracefully stop background processes on script exit

### `permissions` (`astral-sh__uv`)

````yaml
permissions: {}
````

> **concept:** how to disable all permission checks in the config

### `_recover_pending_embeds` (`rbtr__rbtr`)

````python
def _recover_pending_embeds(self, store: IndexStore) -> None:
        """Check for indexed commits with un-embedded chunks and wake the worker.

        Handles the case where the daemon crashed after indexing
        but before embedding completed.  Sets `_wake` so the
        DB-polling worker picks up the work.
        """
        for repo_id, _repo_path in store.list_repos():
            for sha, _ts in store.list_indexed_commits(repo_id):
                count = store.count_unembedded(repo_id, sha)
                if count > 0:
                    log.info(
                        "Recovering embed for repo_id=%d sha=%s (%d chunks)",
                        repo_id,
                        sha[:12],
                        count,
                    )
                    self._wake.set()
                    return
````

> **concept:** find indexed commits with chunks missing embeddings after a crash and
> trigger reprocessing
