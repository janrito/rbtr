/**
 * Echo the arguments a tool actually received.
 *
 * When a call returns nothing useful, appending the received arguments
 * to the message lets the model diagnose a mis-shaped argument *from
 * context* — a stringified array, a wrong ref, a typo'd path — without
 * the tool guessing at any specific failure mode. General transparency,
 * not error-pattern fishing.
 */
export function echoArgs(args: Record<string, unknown>, keys: readonly string[]): string {
  const shown = keys.filter((key) => args[key] !== undefined).map((key) => `${key}=${JSON.stringify(args[key])}`);
  return shown.length === 0 ? "" : `\n\nArguments received: ${shown.join(", ")}`;
}

function parseJsonStringArray(text: string): string[] | null {
  const trimmed = text.trim();
  if (!trimmed.startsWith("[")) return null;
  try {
    const parsed: unknown = JSON.parse(trimmed);
    if (Array.isArray(parsed) && parsed.every((item) => typeof item === "string")) {
      return parsed as string[];
    }
  } catch {
    // not JSON — fall through
  }
  return null;
}

/**
 * Read a `string[]` argument, tolerating the JSON-encoded shapes a
 * provider may deliver (a bare `'["a"]'` string, or a one-element list
 * wrapping it) — the same shapes the daemon decodes. So both display
 * (call lines) and extension-side logic see the real values regardless
 * of wire encoding.
 */
export function decodeStringList(value: unknown): string[] {
  if (typeof value === "string") return parseJsonStringArray(value) ?? [value];
  if (Array.isArray(value)) {
    if (value.length === 1 && typeof value[0] === "string") {
      const decoded = parseJsonStringArray(value[0]);
      if (decoded) return decoded;
    }
    return value.filter((item): item is string => typeof item === "string");
  }
  return [];
}
