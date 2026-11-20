# A base whose members a subclass reaches without naming `self`.
class Model
  def save
    check()
  end

  def check
    true
  end
end
