// Greeter — format greetings for named recipients.
//
// The JavaScript plugin extracts function declarations, generator
// functions, arrow functions bound to consts, classes, module-level
// const/let variables (including destructuring), imports, and methods
// (class members, including get/set accessors; object-literal methods are
// captured without a scope).

import { LOCALE } from "./config.js";
import format from "formatter";
import "./styles.css";

export const DEFAULT_GREETING = "Hello";
const { locale, fallback } = LOCALE;

/** Format a greeting for a name. */
export function formatGreeting(name) {
  return `${DEFAULT_GREETING}, ${name} (${locale ?? fallback})`;
}

const cachedDefault = () => DEFAULT_GREETING;

/** Stateful greeter holding a prefix. */
export class Greeter {
  constructor(prefix = DEFAULT_GREETING) {
    this.prefix = prefix;
  }

  greet(name) {
    return format(`${this.prefix}, ${name}`);
  }
}

/** Yield each recipient name in turn. */
export function* recipients(names) {
  for (const name of names) {
    yield name;
  }
}

/** Greeting helpers grouped as an object literal. */
export const helpers = {
  shout(name) {
    return `${name.toUpperCase()}!`;
  },
};
