/* Header companion for c.c — declares the greeter API.
 *
 * Resolves the `#include "greeter.h"` edge and exercises C prototype and
 * type-definition capture (declared, not defined, here). */
#ifndef GREETER_H
#define GREETER_H

#define MAX_NAME 64

/* A greeter holding a prefix string. */
struct Greeter {
    const char *prefix;
};

typedef struct Greeter Greeter;

/* Build a greeter with the default prefix. */
Greeter greeter_default(void);

/* Format a greeting for `name` into `buf`. */
int format_greeting(const Greeter *g, const char *name, char *buf, int n);

#endif
