# Adjust the files accordingly if you are not using the machines of polytechnique
EIGEN=/usr/local/eigen-3.4.0
ANN_INCLUDE=/usr/local/ann-1.1.2/include
ANN_LIB=/usr/local/ann-1.1.2/lib

CXX = g++

INCLUDES=-I$(EIGEN) -I$(ANN_INCLUDE) -IDataset -IRegression -IRegression/KnnRegression -IRegression/LinearRegression
CFLAGS = -std=c++11 -O2 -Wall
CXXFLAGS = ${INCLUDES} ${CFLAGS}

.PHONY: all clean

all: grader

# TD-specific part

SOURCES_SPECIFIC = Dataset/Dataset.cpp Regression/KnnRegression/KnnRegression.cpp Regression/LinearRegression/LinearRegression.cpp Regression.cpp
OBJECTS_SPECIFIC = Dataset.o KnnRegression.o LinearRegression.o Regression.o

Dataset.o: Dataset/Dataset.cpp Dataset/Dataset.hpp
	$(CXX) $(CFLAGS) -c Dataset/Dataset.cpp

Regression.o: Regression/Regression.cpp Regression/Regression.hpp
	$(CXX) $(CXXFLAGS) -c Regression/Regression.cpp

KnnRegression.o: Regression/KnnRegression/KnnRegression.cpp Regression/KnnRegression/KnnRegression.hpp
	$(CXX) $(CXXFLAGS) -c Regression/KnnRegression/KnnRegression.cpp

LinearRegression.o: Regression/LinearRegression/LinearRegression.cpp Regression/LinearRegression/LinearRegression.hpp
	$(CXX) $(CXXFLAGS) -c Regression/LinearRegression/LinearRegression.cpp

# Common part

SOURCES_COMMON = gradinglib/gradinglib.cpp grading/grading.cpp main.cpp
OBJECTS_COMMON = gradinglib.o grading.o main.o

grader: $(OBJECTS_COMMON) $(OBJECTS_SPECIFIC) 
	$(CXX) $(CXXFLAGS) -o grader $(OBJECTS_COMMON) $(OBJECTS_SPECIFIC)  -L$(ANN_LIB) -lANN

gradinglib.o: gradinglib/gradinglib.cpp gradinglib/gradinglib.hpp
	$(CXX) -c $(CXXFLAGS) -o gradinglib.o gradinglib/gradinglib.cpp

grading.o: grading/grading.cpp gradinglib/gradinglib.hpp
	$(CXX) -c $(CXXFLAGS) -o grading.o grading/grading.cpp 

main.o: main.cpp grading/grading.hpp
	$(CXX) -c $(CXXFLAGS) -o main.o main.cpp

test_k_knn: test_k_knn.o KnnRegression.o LinearRegression.o Regression.o Dataset.o
	$(CXX)  -o test_k_knn test_k_knn.o $(OBJECTS_SPECIFIC)  -L$(ANN_LIB) -lANN

clean:
	rm -f grader *.o test_k_knn
	rm -f *~ output.txt
