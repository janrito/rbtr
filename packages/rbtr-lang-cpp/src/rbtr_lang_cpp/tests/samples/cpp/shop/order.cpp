#include "order.hpp"

namespace shop {

bool Order::save() {
    Model::save();
    return this->check();
}

}  // namespace shop
