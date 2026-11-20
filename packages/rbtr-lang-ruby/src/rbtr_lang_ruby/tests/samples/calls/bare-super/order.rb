class Model
  def save
    true
  end
end

class Order < Model
  def save
    super
  end
end
