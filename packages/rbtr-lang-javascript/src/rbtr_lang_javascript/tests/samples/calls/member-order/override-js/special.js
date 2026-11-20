import { Order } from "./order.js";

export class Special extends Order {
  run() {
    return this.save() && this.check();
  }
}
