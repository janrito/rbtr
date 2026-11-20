require_relative "base"
require_relative "mixin"

class C < Base
  include Mixin

  def run
    self.m
  end
end
