CXX = g++
CFLAGS = -std=c++11 -O2

# TD-specific part

OBJECTS_SPECIFIC = point.o cloud.o edge.o graph.o dendrogram.o
SOURCES_SPECIFIC = point.cpp cloud.cpp edge.cpp graph.cpp dendrogram.cpp

point.o: point.cpp
	$(CXX) -c $(CFLAGS) point.cpp

cloud.o: cloud.cpp
	$(CXX) -c $(CFLAGS) cloud.cpp
	
edge.o: edge.cpp
	$(CXX) -c $(CFLAGS) edge.cpp	

graph.o: graph.cpp
	$(CXX) -c $(CFLAGS) graph.cpp	

dendrogram.o: dendrogram.cpp
	$(CXX) -c -$(CFLAGS) dendrogram.cpp

# Common part

SOURCES_COMMON = gradinglib/gradinglib.cpp grading/grading.cpp grader.cpp
OBJECTS_COMMON = gradinglib.o grading.o grader.o

grader: $(OBJECTS_COMMON) $(OBJECTS_SPECIFIC) 
	$(CXX) $(CFLAGS) -o grader $(OBJECTS_COMMON) $(OBJECTS_SPECIFIC)

gradinglib.o: gradinglib/gradinglib.cpp gradinglib/gradinglib.hpp
	$(CXX) -c $(CFLAGS) -o gradinglib.o gradinglib/gradinglib.cpp

grading.o: grading/grading.cpp gradinglib/gradinglib.hpp
	$(CXX) -c $(CFLAGS) -o grading.o grading/grading.cpp

grader.o: grader.cpp
	$(CXX) -c $(CFLAGS) -o grader.o grader.cpp

test-dendrogram.o: test-dendrogram.cpp
	$(CXX) -c $(CFLAGS) -o test-dendrogram.o test-dendrogram.cpp

test-dendrogram: test-dendrogram.o $(OBJECTS_SPECIFIC)
	$(CXX) $(CFLAGS) -o test-dendrogram $(OBJECTS_SPECIFIC) test-dendrogram.o

.PHONY: all clean

clean:
	rm -f *.o
	rm -f grader
	rm -f test-dendrogram
