package specific;

public interface J extends I {
  default String m() {
    return "j";
  }
}
