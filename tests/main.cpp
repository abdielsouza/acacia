#include <iostream>
#include <cstdlib>

// Forward declarations for test functions
void run_ml_tests();
void run_cv_tests();

int main(int argc, char const *argv[])
{
    std::cout << "Running Acacia Library Tests..." << std::endl;

    int result = 0;

    // Run ML tests
    std::cout << "\n=== Running ML Tests ===" << std::endl;
    try {
        run_ml_tests();
        std::cout << "ML tests completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ML tests failed: " << e.what() << std::endl;
        result = 1;
    }

    // Run CV tests
    std::cout << "\n=== Running CV Tests ===" << std::endl;
    try {
        run_cv_tests();
        std::cout << "CV tests completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "CV tests failed: " << e.what() << std::endl;
        result = 1;
    }

    if (result == 0) {
        std::cout << "\nAll tests passed!" << std::endl;
    } else {
        std::cout << "\nSome tests failed!" << std::endl;
    }

    return result;
}
