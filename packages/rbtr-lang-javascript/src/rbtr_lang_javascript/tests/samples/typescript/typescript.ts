// Greeter — format greetings for named recipients.
//
// The TypeScript plugin extracts function declarations, arrow functions
// bound to consts, classes, module variables, and imports, with
// namespaces forming a scope. It also captures interfaces, enums, type
// aliases, and abstract classes (all as classes), and class/interface
// members as methods, including get/set accessors and abstract method
// signatures.

import { LOCALE } from "./config";
import type { Formatter } from "./types";

export const DEFAULT_GREETING: string = "Hello";

/** Tone of a greeting. */
export enum Tone {
  Formal,
  Casual,
}

/** A function that formats a greeting for a name. */
export type GreetingFn = (name: string) => string;

/** Format a greeting for a name. */
export function formatGreeting(name: string): string {
  return `${DEFAULT_GREETING}, ${name} (${LOCALE})`;
}

const cachedDefault = (): string => DEFAULT_GREETING;

/** A greeting formatter contract. */
export interface Greeting {
  text: string;
  format: Formatter;
  render(name: string): string;
}

/** Base greeter defining the prefix contract. */
abstract class AbstractGreeter {
  protected x: string = DEFAULT_GREETING;

  abstract greet(name: string): string;

  get prefix(): string {
    return this.x;
  }

  set prefix(value: string) {
    this.x = value;
  }
}

/** Stateful greeter holding a prefix. */
export class Greeter extends AbstractGreeter {
  greet(name: string): string {
    return `${this.x}, ${name}`;
  }
}

namespace util {
  export function trim(value: string): string {
    return value.trim();
  }
}
