# Adjust the files accordingly if you are not using the machines of polytechnique
LIBSVM=libsvm-3.23

CXX = g++
CXXFLAGS = ${INCLUDES} -std=c++11 -O2

.PHONY: all clean

all: grader test_perceptron

# TD-specific part

SOURCES_SPECIFIC = activation.cpp dataset.cpp node.cpp neuron.cpp perceptron.cpp
OBJECTS_SPECIFIC = activation.o dataset.o node.o neuron.o perceptron.o

%: %.o
	$(CXX)  $(CXXFLAGS) $^ $(LDFLAGS) -o $@

activation.o:	activation.cpp activation.hpp
dataset.o:	dataset.cpp dataset.hpp
node.o:		node.cpp node.hpp
neuron.o:	neuron.cpp neuron.hpp node.hpp
perceptron.o:	perceptron.cpp perceptron.hpp neuron.hpp dataset.hpp

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

test_perceptron : test_perceptron.cpp perceptron.o neuron.o node.o dataset.o activation.o
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f grader *.o
	rm -f *~ output.txt

