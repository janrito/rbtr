#pragma once

#include "model.hpp"

namespace shop {

class Order : public Model {
public:
    bool save();
};

}  // namespace shop
