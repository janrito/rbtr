# frozen_string_literal: true

# Greeter — format greetings for named recipients.
#
# The Ruby plugin extracts top-level defs (as functions), classes and
# modules, defs inside them (as methods), constant assignments (as
# variables, scoped to their class/module), the RSpec describe/it DSL
# (groups as classes, examples as functions), and require /
# require_relative imports.

require "json"
require_relative "./config"

DEFAULT_GREETING = "Hello"

# Mixin providing a shout helper.
module Shoutable
  def shout(message)
    message.upcase
  end
end

# Formats greetings with a prefix.
class Greeter
  include Shoutable

  MAX_NAME_LENGTH = 64

  def initialize(prefix = DEFAULT_GREETING)
    @prefix = prefix
  end

  # Greet a single recipient.
  def greet(name)
    "#{@prefix}, #{name}"
  end

  def self.default
    new(DEFAULT_GREETING)
  end
end

# Top-level helper that builds a greeter and greets.
def format_greeting(name)
  Greeter.new.greet(name)
end

# RSpec-style specification exercising the describe/it DSL.
RSpec.describe Greeter do
  describe "#greet" do
    it "includes the configured prefix" do
      expect(Greeter.new.greet("Sam")).to include("Hello")
    end
  end
end
