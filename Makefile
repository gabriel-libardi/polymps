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
SOURCE_DIR     := src
INCLUDE_DIR    := include

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
CPPFLAGS ?=  $(addprefix -I$(INCLUDE_DIR)/, $(IMPLEMENTATIONS)) 
CPPFLAGS +=  -Ieigen -Ijson/single_include/nlohmann -Ilibigl/include

# Got some linker flags ?
# -L is a linker flag
LDFLAGS := -L $(LIB_DIR)
LDLIBS  := -lstdc++fs

# CUDA compiler definitions
NVCC       := nvcc
CUFLAGS    := -I/mnt/nfs/modules/apps/cuda-toolkit/9.0.176/samples/common/inc
CULDFLAGS  := -lcudart

define find_subdirs
$(shell find $(1) -maxdepth 1 -type d -exec basename {} \; | grep -v $(1))
endef

# Implementations are different versions of PolyMPS made for different computational demands
# An OpenMP and a CUDA cimplementation are available currently
IMPLEMENTATIONS := $(call find_subdirs, $(SOURCE_DIR))

# Directives that are not filenames to be built
.PHONY: all debug $(IMPLEMENTATIONS) clean help tidy

ifneq ($(IMPLEMENTATIONS),$(call find_subdirs, $(INCLUDE_DIR)))
	$(shell echo "\033[0;31mWarning\033[0m: include subtree does not match source code structure")
	$(shell echo 'Check whether the subdirectories "$(IMPLEMENTATIONS)" match what's inside $(INCLUDE_DIR)/')
endif

# Useful commands definitions
MKDIR := mkdir -p
MV    := mv -f
RM    := rm -rf
SED   := sed
TEST  := test

define to_uppercase
$(shell echo $(1) | tr '[:lower:]' '[:upper:]')
endef

# Creates build directories if they do not exist
define build_dirs
	$(eval UPPER_NAME := $(call to_uppercase, $(1)))
	$(eval OBJECT_$(UPPER_NAME)_DIR := $(OBJECT_DIR)/$(1))
	$(eval INCLUDE_$(UPPER_NAME) := $(INCLUDE_DIR)/$(1))
	$(eval SOURCE_$(UPPER_NAME) := $(SOURCE_DIR)/$(1))

	$(eval dirs_$(1) := $(BINARY_DIR)/ $$(OBJECT_$(UPPER_NAME)_DIR)/)
	$(eval source_$(1) := $$(wildcard $$(SOURCE_$(UPPER_NAME))/*.cpp))
	$(eval target_objects_$(1) := $$(addprefix $$(OBJECT_$(UPPER_NAME)_DIR)/, $$(notdir $$(TARGET_$(UPPER_NAME):.cpp=.o))))
	$(eval lib_objects_$(1) := $$(addprefix $$(OBJECT_$(UPPER_NAME)_DIR)/, $$(notdir $$(source_$(1):.cpp=.o))))
	$(eval objects_$(1) := $$(target_objects_$(1)) $$(lib_objects_$(1)))
	$(eval dependencies_$(1) := $$(objects_$(1):.o=.d))
	$(eval targets_$(1) := $$(addprefix $$(BINARY_DIR)/, $$(notdir $$(target_objects_$(1):.o=))))
endef

# Build all implementations
all: $(IMPLEMENTATIONS)

# Compile with helpful warnings (-Wall -Wextra flags)
debug: CXXFLAGS += $(WARNINGS) $(DEBUG)
debug: clean
debug: $(IMPLEMENTATIONS)

# Rule for creating directories
%/:
	@$(MKDIR) $@

# Rule for each implementation
.SECONDEXPANSION:
$(foreach impl, $(IMPLEMENTATIONS), $(eval $(call build_dirs,$(impl))))
$(info target_objects_omp: $(target_objects_omp))
$(info lib_objects_omp: $(lib_objects_omp))
$(info objects_omp: $(objects_omp))
$(info dependencies_omp: $(dependencies_omp))
$(IMPLEMENTATIONS): $$(dirs_$$@) $$(targets_$$@)
	@echo "Built $(call to_uppercase, $@) implementation successfully"

# List the prerequisites for building your executable, and fill its
# recipe to tell make what to do with these
$(BINARY_DIR)/%: $(OBJECT_OMP_DIR)/%.o $(lib_objects_omp)
	$(LINK.cpp) $^ $(LOADLIBES) $(LDLIBS) -o $@

# Since your source and object files don't share the same prefix, you
# need to tell make exactly what to do since its built-in rules don't
# cover your specific case
.SECONDEXPANSION:
$(OBJECT_OMP_DIR)/%.o: $(SOURCE_OMP)/%.cpp $$(wildcard $$(INCLUDE_OMP)/*.h)
	$(COMPILE.cpp) $(OUTPUT_OPTION) $<

.SECONDARY: $(objects_omp)


clean:
	$(RM) $(OBJECT_DIR) $(BINARY_DIR)


ifneq "$(MAKECMDGOALS)" "clean"
	-include $(dependencies_omp)
endif


help:
	@echo 'Build all source files.'
	@echo
	@echo 'Targets in this file are:'
	@echo 'all     Compile and link all source files.'
	@echo 'clean   Remove all intermediate files.'
	@echo 'help    Display this information.'
