#include <iostream>
#include "grading/grading.hpp"

using namespace std;

#define GRADING 1

int main(int argc, char* argv[])
{
#if GRADING != 1
    // START OF THE CUSTOM CODE SECTION
    // This code will be executed only if you set GRADING to a value different from 1

    // END OF THE CUSTOM CODE SECTION
#else
    // RUN THE AUTOMATIC GRADER
    {
      int test_number = 0; // run all the tests

      if (argc == 2) {
        test_number = stoi(argv[1]);
      }

      return tdgrading::grading(std::cerr, test_number);
      // END OF THE AUTOMATIC GRADER
    }
#endif
}
