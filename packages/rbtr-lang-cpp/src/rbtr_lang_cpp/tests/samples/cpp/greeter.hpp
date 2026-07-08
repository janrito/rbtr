// Header companion for cpp.cpp — declares the greeter API.
//
// Exercises in-class *declared-only* methods (no inline body): the
// constructor and member functions are captured as methods scoped to
// their class, just like inline definitions.
#pragma once

#include <string>

namespace greet {

class Greeting {
public:
    Greeting(std::string prefix);
    std::string render(const std::string &name) const;
    static Greeting withDefault();

private:
    std::string prefix_;
};

}  // namespace greet
