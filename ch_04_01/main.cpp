#if(UNIT_TEST)
#define BOOST_TEST_MAIN
//#define BOOST_TEST_DYN_LINK

//#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_CASE(TEST_main)
{
    std::cout << "main\n";
}

#else
#include <iostream>

int main(int argc, const char* argv[])
{
    std::cout << "hello world\n";
    return 0;
}

#endif // UNIT_TEST
