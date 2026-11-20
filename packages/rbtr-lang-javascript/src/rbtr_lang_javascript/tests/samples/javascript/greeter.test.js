// Tests of the greeter: each `describe`, `it` and `test` is a test, named
// by its first argument, and one inside a group is nested in it.

import { formatGreeting } from "./javascript.js";

describe("formatGreeting", () => {
  function expected(name) {
    return `Hello, ${name}`;
  }

  it("greets by name", () => {
    // The locale is the module's default.
    expect(formatGreeting("Ada")).toContain(expected("Ada"));
  });
});

test("greets nobody", () => {
  expect(formatGreeting("")).toContain("Hello");
});

if (process.env.CI) {
  console.log("on CI");
}
