CXX ?= g++
COMPILE_FLAGS = -std=c++17 -Wall -Wextra -pipe 
DEBUG_FLAGS = -ggdb3 -gdwarf-5 

SRC_FILES = network.cpp load_data.cpp 
HEADERS = network.hpp load_data.hpp

all: train test

debug: train-debug test-debug

lines:
	wc -l $(SRC_FILES) $(HEADERS) test.cpp train.cpp Makefile

train: $(HEADERS) $(SRC_FILES) train.cpp
	$(CXX) $(COMPILE_FLAGS) -O3 $(SRC_FILES) train.cpp -o train

test: $(HEADERS) $(SRC_FILES) test.cpp
	$(CXX) $(COMPILE_FLAGS) -O3 $(SRC_FILES) test.cpp -o test

train-debug: $(HEADERS) $(SRC_FILES) train.cpp
	$(CXX) $(COMPILE_FLAGS) $(DEBUG_FLAGS) $(SRC_FILES) train.cpp -o train-debug

test-debug: $(HEADERS) $(SRC_FILES) test.cpp
	$(CXX) $(COMPILE_FLAGS) $(DEBUG_FLAGS) $(SRC_FILES) test.cpp -o test-debug

network.o: network.hpp network.cpp .o/
	$(CXX) $(COMPILE_FLAGS) -O3 -c network.cpp -o .o/


.o/:
	mkdir .o/

clean: 
	rm train test train-debug test-debug

clear:
	rm train test train-debug test-debug
