// Greeter — format greetings for named recipients.
//
// The Java plugin extracts classes, methods (including constructors),
// fields (as variables), and imports. Interfaces, enums, and records are
// captured as classes, and their members scope to them.
package com.example.greet;

import java.util.List;
import static java.util.Collections.emptyList;

import com.example.format.Formatter;

// Standalone note at top level, separated by blank lines.
// Second line of the same block.

/** Formats greetings with a configurable prefix. */
public class Greeter implements Formatter {
    private final String prefix;

    public Greeter(String prefix) {
        this.prefix = prefix;
    }

    /** Format a greeting (implements the Formatter contract). */
    @Override
    public String format(String name) {
        return greet(name);
    }

    /** Greet a single recipient. */
    public String greet(String name) {
        return prefix + ", " + name;
    }

    /** Build a greeter with the default prefix. */
    public static Greeter withDefault() {
        return new Greeter("Hello");
    }

    /** A nested formatter. */
    static class Formatter {
        String format(String name) {
            return name.trim();
        }
    }
}

/** A greeting strategy contract. */
interface Strategy {
    String render(String name);
}

/** Supported greeting tones. */
enum Tone {
    FORMAL,
    CASUAL;

    /** Human-readable label for this tone. */
    String label() {
        return name().toLowerCase();
    }
}

/** An immutable named recipient. */
record Recipient(String name, String locale) {
    /** Full display name including locale. */
    String display() {
        return name + " (" + locale + ")";
    }
}

/** Marks a greeter method as audited. */
@interface Audited {
    String value();
}
