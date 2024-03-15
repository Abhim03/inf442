// Include the library that defines input/output streams and operations on them
#include <iostream>

int main()  {
   // Declare the variable 'name' to be an array of 256 characters 
  char name[256];
  
  

  int count = 0;
  
  std::cout << "Give a positive number: ";
  std::cin >> count;

  std::cout << "Hey";
  for (int i = 1; i < count; i++) {   // <-- Here goes the loop!
    std::cout << "-hey";
  }

  std::cout << ", " << name << "!" << std::endl;

  return 0;
}