#pragma once
#include "order.hpp"

class Special : public Order {
 public:
  bool run() { return save() && check(); }
};
