require_relative "a"
require_relative "b"

# The multi case.
class C
  include A, B

  def run
    self.m
  end
end
