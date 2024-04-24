EIGEN=/usr/local/eigen-3.4.0
ANN_INCLUDE=/usr/local/ann-1.1.2/include
ANN_LIB=/usr/local/ann-1.1.2/lib

all: test_dataset test_knn test_random_projection test_roc_curve

CXX = g++
CFLAGS = -std=c++11 -O2

# TD-specific part

OBJECTS_SPECIFIC = Dataset.o Classification.o KnnClassification.o ConfusionMatrix.o RandomProjection.o
SOURCES_SPECIFIC = Dataset.cpp Classification.cpp KnnClassification.cpp ConfusionMatrix.cpp RandomProjection.cpp

Dataset.o: Dataset.cpp
	$(CXX) -c $(CFLAGS) Dataset.cpp

Classification.o: Classification.cpp
	$(CXX) -c $(CFLAGS) -I$(ANN_INCLUDE) Classification.cpp
	
KnnClassification.o: KnnClassification.cpp
	$(CXX) -c $(CFLAGS) -I$(ANN_INCLUDE) KnnClassification.cpp	

ConfusionMatrix.o: ConfusionMatrix.cpp
	$(CXX) -c $(CFLAGS) -I$(ANN_INCLUDE) ConfusionMatrix.cpp	

RandomProjection.o: RandomProjection.cpp
	$(CXX) -c -$(CFLAGS) -I$(EIGEN) RandomProjection.cpp	

test_dataset: test_dataset.cpp Dataset.o
	$(CXX) $(CFLAGS) -I$(EIGEN) test_dataset.cpp Dataset.o -o test_dataset

test_knn: test_knn.cpp ConfusionMatrix.o KnnClassification.o Classification.o Dataset.o
	$(CXX) $(CFLAGS) -I$(EIGEN) -I$(ANN_INCLUDE) test_knn.cpp ConfusionMatrix.o KnnClassification.o Classification.o Dataset.o -o test_knn -L$(ANN_LIB) -lANN 

test_random_projection: test_random_projection.cpp RandomProjection.o ConfusionMatrix.o KnnClassification.o Classification.o Dataset.o
	$(CXX) $(CFLAGS) -I$(EIGEN) -I$(ANN_INCLUDE) test_random_projection.cpp RandomProjection.o ConfusionMatrix.o KnnClassification.o Classification.o Dataset.o -o test_random_projection -L$(ANN_LIB) -lANN

test_roc_curve: test_roc_curve.cpp ConfusionMatrix.o KnnClassification.o Classification.o Dataset.o
	$(CXX) $(CFLAGS) -I$(EIGEN) -I$(ANN_INCLUDE) test_roc_curve.cpp ConfusionMatrix.o KnnClassification.o Classification.o Dataset.o -o test_roc_curve -L$(ANN_LIB) -lANN

# Common part

SOURCES_COMMON = gradinglib/gradinglib.cpp grading/grading.cpp main.cpp
OBJECTS_COMMON = gradinglib.o grading.o main.o

grader: $(OBJECTS_COMMON) $(OBJECTS_SPECIFIC) 
	$(CXX) $(CFLAGS) -o grader $(OBJECTS_COMMON) $(OBJECTS_SPECIFIC) -I$(EIGEN) -I$(ANN_INCLUDE) -L$(ANN_LIB) -lANN

gradinglib.o: gradinglib/gradinglib.cpp gradinglib/gradinglib.hpp
	$(CXX) -c $(CFLAGS) -o gradinglib.o gradinglib/gradinglib.cpp

grading.o: grading/grading.cpp gradinglib/gradinglib.hpp
	$(CXX) -c $(CFLAGS) -o grading.o grading/grading.cpp -I$(ANN_INCLUDE) -I$(EIGEN)

main.o: main.cpp grading/grading.hpp
	$(CXX) -c $(CFLAGS) -o main.o main.cpp


.PHONY: all clean

clean:
	rm -f *.o
	rm -f test_dataset
	rm -f test_1q2
	rm -f test_2q1
	rm -f test_3q1
	rm -f test_3q2
	rm -f test_knn
	rm -f test_random_projection
	rm -f test_roc_curve
	rm -f grader
