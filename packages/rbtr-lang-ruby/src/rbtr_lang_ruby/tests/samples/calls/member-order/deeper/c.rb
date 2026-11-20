require_relative "a"
require_relative "b"

# The deeper case.
class C
  include A
  include B

  def run
    self.m
  end
end
