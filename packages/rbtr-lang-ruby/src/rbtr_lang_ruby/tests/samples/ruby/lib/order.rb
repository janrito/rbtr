require_relative "model"

# A subclass of a required base.
class Order < Model
  def save
    check() && self.check
  end

  def self.build
    Order.new
  end
end
