/* Includes its own sibling by bare name. */
#include "text.h"

int text_length(const char *s) {
  int n = 0;
  while (s[n] != '\0') {
    n++;
  }
  return n;
}
