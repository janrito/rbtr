// Orders — a typed base and its subclass; `implements` names no base.

interface Saveable {
  save(): boolean;
}

class Model implements Saveable {
  save(): boolean {
    return this.check<boolean>();
  }

  check<T>(): T {
    return true as T;
  }
}

export class Order extends Model {
  save(): boolean {
    return super.save();
  }
}
