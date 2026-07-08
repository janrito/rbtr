# rbtr-lang-java

Java support for [rbtr](../rbtr). Optional plugin — install with
`pip install rbtr[java]`.

## What it ingests

Java is class-only (no free functions), so extraction is class-centred:

- **Classes** — classes, interfaces, enums, records, annotations.
- **Methods** — methods and constructors, scoped to their type.
- **Variables** — fields (class members are data-scope, captured as
  variables) and enum constants.
- **Imports** — `import` (and `import static`) declarations.

Leading Javadoc (`/** … */`) folds into the symbol's content.

## Chunks produced

```java
public class Formatter {           // class "Formatter"
  private int width;               //   variable "width", scope "Formatter"
  Formatter(int w) { … }           //   method "Formatter", scope "Formatter"
  String format(String s) { … }    //   method "format", scope "Formatter"
}
import java.util.List;             // import, metadata {module: java.util.List}
```

## Embedded / injected chunks

None. Java does not embed other languages.

## Grammar & dependencies

Uses the `tree-sitter-java` grammar. No dependency on other language plugins.
