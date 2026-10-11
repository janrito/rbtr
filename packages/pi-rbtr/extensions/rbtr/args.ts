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

/**
 * A tool's `prepareArguments`: each of *keys* a provider sent as a
 * JSON-encoded string, or a list wrapping one, becomes the list it
 * encodes. Runs before pi validates the call, which checks the result
 * against the tool's schema; other arguments pass through untouched.
 */
export function prepareStringLists<T>(...keys: string[]): (args: unknown) => T {
  return (args) => {
    if (typeof args !== "object" || args === null || Array.isArray(args)) return args as T;
    const prepared: Record<string, unknown> = { ...args };
    for (const key of keys) {
      const value = prepared[key];
      if (typeof value === "string" || Array.isArray(value)) prepared[key] = decodeStringList(value);
    }
    return prepared as T;
  };
}

/**
 * The refs a slash command names: each whitespace-separated word of its
 * arguments. None when it names none, so the caller's default applies.
 */
export function commandRefs(args: string): string[] {
  return args.split(/\s+/).filter((word) => word.length > 0);
}
