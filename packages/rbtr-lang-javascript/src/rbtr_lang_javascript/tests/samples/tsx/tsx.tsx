// Counter — a small React component rendering a clickable count.
//
// The tsx plugin uses the `language_tsx` grammar, not
// `language_typescript`, which cannot parse JSX (a `.tsx` file parses
// with errors under the plain TypeScript grammar). It runs the same
// query as the typescript plugin against the JSX-aware grammar, so it
// extracts function declarations, arrow-function consts, module
// variables, imports, and interfaces (as classes — the CounterProps
// interface below).

import { useState } from "react";
import type { ReactNode } from "react";
import { INITIAL_LABEL } from "./labels";

export const INITIAL_COUNT: number = 0;

/** Props for the Counter component. */
interface CounterProps {
  label: string;
}

/** A button that increments a counter on each click. */
export function Counter({ label }: CounterProps): ReactNode {
  const [count, setCount] = useState(INITIAL_COUNT);
  return (
    <button onClick={() => setCount(count + 1)}>
      {label}: {count}
    </button>
  );
}

/** Render a static badge as a span. */
const Badge = (text: string): ReactNode => <span className="badge">{text}</span>;
