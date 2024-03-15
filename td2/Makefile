


CXX = g++
CFLAGS = -std=c++11 -O2

.PHONY: all clean

all: grader

# TD-specific part

OBJECTS_SPECIFIC = retrieval.o 
SOURCES_SPECIFIC = retrieval.cpp 

# Common part

SOURCES_COMMON = gradinglib/gradinglib.cpp grading/grading.cpp main.cpp
OBJECTS_COMMON = gradinglib.o grading.o main.o

grader: $(OBJECTS_COMMON) $(OBJECTS_SPECIFIC) 
	$(CXX) $(CFLAGS) -o grader $(OBJECTS_COMMON) $(OBJECTS_SPECIFIC) 

gradinglib.o: gradinglib/gradinglib.cpp gradinglib/gradinglib.hpp
	$(CXX) -c $(CFLAGS) -o gradinglib.o gradinglib/gradinglib.cpp

grading.o: grading/grading.cpp gradinglib/gradinglib.hpp
	$(CXX) -c $(CFLAGS) -o grading.o grading/grading.cpp 

main.o: main.cpp grading/grading.hpp
	$(CXX) -c $(CFLAGS) -o main.o main.cpp


clean:
	rm -f grader *.o
	rm -f *~