# Build all source files.
#
# Targets in this file are:
# all     Compile and link all source files.
# clean   Remove all intermediate files.
# help    Display this information.
#
# Copyright (c) 2021 Rubens AMARO
# Distributed under the MIT License.

# Directories
BINARY_DIR     := bin
OBJECT_DIR     := obj
OBJECT_LIB_DIR := $(OBJECT_DIR)/lib
INCLUDE_DIR    := include
LIB_DIR        := lib

# CUDA-specific implementation paths and targets
SOURCE_CUDA    := gpu
TARGET_CUDA    := emps_cuda.cu

# OpenMP-specific implementation paths and targets
SOURCE_OMP     := src
TARGET_OMP     := main.cpp

# C compiler (GNU/INTEL) (g++) or (icpc)
CC          := g++
CXX         := g++
DEBUG       := -g -fvar-tracking
PERFORMANCE := -Ofast -mtune=native -march=native
WARNINGS    := -Wall -Wextra

# Got some preprocessor flags to pass ?
# -I is a preprocessor flag, not a compiler flag

CXXFLAGS :=  $(PERFORMANCE) -std=c++11 -MP -MMD -fopenmp -lm -D_GNU_SOURCE
CPPFLAGS := -I $(INCLUDE_DIR) -Ieigen -Ijson/single_include/nlohmann -Ilibigl/include

# Got some linker flags ?
# -L is a linker flag
LDFLAGS := -L $(LIB_DIR)
LDLIBS  := -lstdc++fs

MKDIR := mkdir -p
MV    := mv -f
RM    := rm -rf
SED   := sed
TEST  := test

# Creates an object directory if it does not exist.
create_binary_directory         := $(shell for f in $(BINARY_DIR); do $(TEST) -d $$f | $(MKDIR) $$f; done)
create_object_directory         := $(shell for f in $(OBJECT_DIR); do $(TEST) -d $$f | $(MKDIR) $$f; done)
create_object_library_directory := $(shell for f in $(OBJECT_LIB_DIR); do $(TEST) -d $$f | $(MKDIR) $$f; done)

# OpenMP implementation's objects and executables
source_omp         := $(wildcard $(SOURCE_OMP)/*.cpp)
target_objects_omp := $(addprefix $(OBJECT_DIR)/, $(notdir $(TARGET_OMP:.cpp=.o)))
lib_objects_omp    := $(addprefix $(OBJECT_LIB_DIR)/, $(notdir $(source_omp:.cpp=.o)))
objects_omp        := $(target_objects_omp) $(lib_objects_omp)
dependencies_omp   := $(objects_omp:.o=.d)
targets_omp        := $(addprefix $(BINARY_DIR)/, $(notdir $(target_objects_omp:.o=)))


# You should indicate whenever a rule does not produce any target output
# with the .PHONY sepcial rule
.PHONY: all debug omp gpu
all: omp gpu

# Compile with helpful warnings (-Wall -Wextra flags)
debug: $(targets_omp) $(targets_gpu)
	CXXFLAGS += $(WARNINGS) $(DEBUG)

# Build OpenMP implementation
omp: $(targets_omp)

# Build CUDA implementation
gpu: $(targets_gpu)

# List the prerequisites for building your executable, and fill its
# recipe to tell make what to do with these
$(BINARY_DIR)/%: $(OBJECT_LIB_DIR)/%.o $(lib_objects_omp)
	$(LINK.cpp) $^ $(LOADLIBES) $(LDLIBS) -o $@

# Since your source and object files don't share the same prefix, you
# need to tell make exactly what to do since its built-in rules don't
# cover your specific case
.SECONDEXPANSION:
$(OBJECT_LIB_DIR)/%.o: $(SOURCE_OMP)/%.cpp $$(wildcard $$(INCLUDE_DIR)/*.h)
	$(COMPILE.cpp) $(OUTPUT_OPTION) $<

.PHONY: clean
clean:
	$(RM) $(OBJECT_DIR) $(BINARY_DIR)

ifneq "$(MAKECMDGOALS)" "clean"
	-include $(dependencies_omp)
endif

.SECONDARY: $(objects_omp)

.PHONY: help
help:
	@echo 'Build all source files.'
	@echo
	@echo 'Targets in this file are:'
	@echo 'all     Compile and link all source files.'
	@echo 'clean   Remove all intermediate files.'
	@echo 'help    Display this information.'
