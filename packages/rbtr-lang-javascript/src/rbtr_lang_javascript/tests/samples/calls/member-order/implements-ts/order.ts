import { Base } from "./base";
import { Saver } from "./saver";

export class Order extends Base implements Saver {
  run(): boolean {
    return this.save();
  }
}
