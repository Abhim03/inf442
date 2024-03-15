CXX = g++
CXXFLAGS = --std=c++11

.PHONY: all clean

all: grader

# TD-specific part

OBJECTS_SPECIFIC = stats_functions.o
SOURCES_SPECIFIC = stats_functions.cpp

stats_functions.o: stats_functions.cpp stats_functions.hpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

# Common part

SOURCES_COMMON = gradinglib/gradinglib.cpp grading/grading.cpp main.cpp
OBJECTS_COMMON = gradinglib.o grading.o main.o

grader: $(OBJECTS_COMMON) $(OBJECTS_SPECIFIC)
	$(CXX) $(CXXFLAGS) -o grader $(OBJECTS_COMMON) $(OBJECTS_SPECIFIC)

gradinglib.o: gradinglib/gradinglib.cpp gradinglib/gradinglib.hpp
	$(CXX) -c $(CXXFLAGS) -o gradinglib.o gradinglib/gradinglib.cpp

grading.o: grading/grading.cpp gradinglib/gradinglib.hpp
	$(CXX) -c $(CXXFLAGS) -o grading.o grading/grading.cpp

main.o: main.cpp grading/grading.hpp
	$(CXX) -c $(CXXFLAGS) -o main.o main.cpp

clean:
	rm -f grader *.o
	rm -f *~
