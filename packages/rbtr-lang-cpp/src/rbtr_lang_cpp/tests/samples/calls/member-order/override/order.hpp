#pragma once
#include "base.hpp"

class Order : public Base {
 public:
  bool save() { return Base::save(); }
};
