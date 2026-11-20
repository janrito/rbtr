require_relative "order"

class Special < Order
  def run
    self.save && self.check
  end
end
