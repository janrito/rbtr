require_relative "left"
require_relative "right"

class Bottom
  include Left
  include Right

  def run
    self.m
  end
end
