#pragma once
#include "left.hpp"
#include "right.hpp"

class Bottom : public Left, public Right {
 public:
  int run() { return m(); }
};
