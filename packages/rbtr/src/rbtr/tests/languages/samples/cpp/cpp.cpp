// Greeter — format greetings for named recipients.
//
// The C++ plugin extracts free functions, prototypes, classes/structs/
// unions, type aliases, namespaces (as classes and scopes), member
// functions and operator overloads (as methods), namespace/global
// variables (including pointer globals), macros, and #include imports.

#include <string>

#include "greeter.hpp"

#define GREET_VERSION 2

const char *kAppName = "greeter";

namespace greet {

const std::string DEFAULT_PREFIX = "Hello";

// A short alias for a recipient name.
using Name = std::string;

// Anything convertible to a std::string can name a recipient.
template <typename T>
concept Nameable = std::convertible_to<T, std::string>;

// A tagged greeting payload.
union Payload {
    int code;
    const char *text;
};

// Build a greeter with the default prefix (declared here).
Greeter make_default();

// A greeter holding a prefix.
class Greeter {
   public:
    explicit Greeter(std::string prefix) : prefix_(std::move(prefix)) {}

    // Greet a single recipient.
    std::string greet(const std::string &name) const {
        return prefix_ + ", " + name;
    }

    // Two greeters are equal when their prefixes match.
    bool operator==(const Greeter &other) const {
        return prefix_ == other.prefix_;
    }

   private:
    std::string prefix_;
};

// Format a greeting using a greeter.
std::string format_greeting(const Greeter &g, const std::string &name) {
    return g.greet(name);
}

}  // namespace greet
