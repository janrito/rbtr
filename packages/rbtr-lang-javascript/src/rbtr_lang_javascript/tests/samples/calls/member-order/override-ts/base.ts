export class Base {
  check(): boolean {
    return true;
  }

  save(): boolean {
    return this.check();
  }
}
