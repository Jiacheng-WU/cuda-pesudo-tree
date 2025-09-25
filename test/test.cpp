#include <boost/ut.hpp>

int main() {
    using namespace boost::ut;
    "sample test"_test = [] { expect(1_i == 1); };
}
