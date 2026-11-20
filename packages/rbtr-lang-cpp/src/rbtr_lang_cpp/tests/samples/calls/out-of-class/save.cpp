#include "order.hpp"

namespace shop {

bool Order::save() { return check(); }

}  // namespace shop

void shop::Order::close() { check(); }
