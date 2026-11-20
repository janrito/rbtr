package override;

public class Special extends Order {
  boolean run() {
    return save() && check();
  }
}
