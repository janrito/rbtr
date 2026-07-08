/* Greeter — format greetings for named recipients.
 *
 * The C plugin extracts functions, function prototypes, struct/union/
 * enum/typedef type definitions (as classes), top-level variables
 * (including pointer-declared globals), function/object-like macros,
 * and #include imports (system and local). */

#include <stdio.h>
#include "greeter.h"

#define MAX_NAME 64
#define SQUARE(x) ((x) * (x))

int greeter_count = 0; // trailing comment: its own chunk, not folded
const char *default_prefix = "Hello";

/* Standalone note, separated by blank lines from any definition. */
/* Second line of the same block. */

/* A greeter holding a prefix string. */
struct Greeter {
    const char *prefix;
};

typedef struct Greeter Greeter;

/* A callback invoked for each formatted greeting. */
typedef void (*GreetCallback)(const char *line);

/* A tagged greeting payload. */
union Payload {
    int code;
    const char *text;
};

enum Locale { LOCALE_EN, LOCALE_FR };

/* Build a greeter with the default prefix. */
Greeter greeter_default(void);

/* Format a greeting for `name` into `buf`. */
int format_greeting(const Greeter *g, const char *name, char *buf, int n) {
    return snprintf(buf, (size_t)n, "%s, %s", g->prefix, name);
}
