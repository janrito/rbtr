// Orders — calls of each shape the call graph resolves, and two it leaves.

import * as greeter from "./javascript.js";
import { formatGreeting as greet } from "./javascript.js";

class Model {
  save() {
    return this.check();
  }

  check() {
    return true;
  }
}

export class Order extends Model {
  save() {
    super.save();
    return new greeter.Greeter().greet(greet("order"));
  }

  total(items) {
    // `sum` is defined nowhere here, and `items[0]` is a value: no edge.
    return sum(items) + Number(this.check()) + items[0].valueOf();
  }
}

export function build() {
  const inner = () => new Order();
  return inner();
}
