CXX = g++
CXXFLAGS = -std=c++11 -O2

.PHONY: all clean

all: grader

# TD-specific part

SOURCES_SPECIFIC = kmeans.cpp 

# Common part

SOURCES_COMMON = gradinglib/gradinglib.cpp grading/grading.cpp main.cpp
OBJECTS_COMMON = gradinglib.o grading.o main.o

grader: $(OBJECTS_COMMON)
	$(CXX) $(CXXFLAGS) -o grader $(OBJECTS_COMMON)

gradinglib.o: gradinglib/gradinglib.cpp gradinglib/gradinglib.hpp
	$(CXX) -c $(CXXFLAGS) -o gradinglib.o gradinglib/gradinglib.cpp

grading.o: grading/grading.cpp gradinglib/gradinglib.hpp $(SOURCES_SPECIFIC)
	$(CXX) -c $(CXXFLAGS) -o grading.o grading/grading.cpp 

main.o: main.cpp grading/grading.hpp
	$(CXX) -c $(CXXFLAGS) -o main.o main.cpp


clean:
	rm -f *.o
	rm -f grader
