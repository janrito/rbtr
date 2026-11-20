import { Order } from "./order";

export class Special extends Order {
  run(): boolean {
    return this.save() && this.check();
  }
}
