// https://github.com/pytorch/examples/blob/master/cpp/dcgan/dcgan.cpp
// http://aidiary.hatenablog.com/entry/20180304/1520172429
//
#if(UNIT_TEST)
#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>
BOOST_AUTO_TEST_CASE(TEST_main)
{
    std::cout << "main\n";
}
#else // UNIT_TEST
#include <iostream>

int main(int argc, const char* argv[])
{
    std::cout << "hello world\n";
    return 0;
}

#endif // UNIT_TEST
