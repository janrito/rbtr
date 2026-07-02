# API

## format-greeting

Formats a greeting for a name.

Call it from Python:

```python
from greeter import format_greeting


def demo() -> str:
    return format_greeting("Ada")
```

Or wrap it in a shell helper:

```sh
greet() {
  format-greeting "$1"
}
```
