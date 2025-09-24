#include <fmt/format.h>
#include <iostream>

int main() {
    std::string message = fmt::format("Hello, {}!", "World");
    std::cout << message << std::endl;
    return 0;
}