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
TARGET_GPU     := main.cu
EXTENSION_GPU  := cu

# OpenMP-specific implementation paths and targets
TARGET_OMP     := main.cpp
EXTENSION_OMP  := cpp

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

# Link OpenMP code to stdc++fs
LDLIBS  := -lstdc++fs

# CUDA compiler definitions
NVCC         := nvcc
CUDAFLAGS    := -I/mnt/nfs/modules/apps/cuda-toolkit/9.0.176/samples/common/inc
CUDAFLAGS    += $(addprefix -I$(INCLUDE_DIR)/, $(IMPLEMENTATIONS))
CUXXFLAGS    := -std=c++11 
CULDLIBS     :=

# Directives that are not filenames to be built
.PHONY: all debug $(IMPLEMENTATIONS) clean help check-structure

define find_subdirs
$(shell find $(1) -maxdepth 1 -type d -exec basename {} \; | grep -v $(1))
endef

# Implementations are different versions of PolyMPS made for different computational demands
# An OpenMP and a CUDA cimplementation are available currently
IMPLEMENTATIONS := $(call find_subdirs, $(SOURCE_DIR))

# Define ANSI color codes
ANSI_RED          := \x1b[31m
ANSI_GREEN        := \x1b[32m
ANSI_LIGHT_YELLOW := \x1b[33m
ANSI_RESET        := \x1b[0m

# Useful commands definitions
MKDIR := mkdir -p
RM    := rm -rf
PRINT := echo -e

define to_uppercase
$(shell $(PRINT) $(1) | tr '[:lower:]' '[:upper:]')
endef

# Creates build directories if they do not exist
define build_dirs
	$(eval UPPER_NAME := $(call to_uppercase, $(1)))
	OBJECT_$(UPPER_NAME)_DIR := $(OBJECT_DIR)/$(1)
	INCLUDE_$(UPPER_NAME)    := $(INCLUDE_DIR)/$(1)
	SOURCE_$(UPPER_NAME)     := $(SOURCE_DIR)/$(1)

	dirs_$(1)           := $(BINARY_DIR)/ $$(OBJECT_$(UPPER_NAME)_DIR)/
	source_$(1)         := $$(wildcard $$(SOURCE_$(UPPER_NAME))/*.$$(EXTENSION_$(UPPER_NAME)))
	target_objects_$(1) := $$(addprefix $$(OBJECT_$(UPPER_NAME)_DIR)/, $$(notdir $$(TARGET_$(UPPER_NAME):.$$(EXTENSION_$(UPPER_NAME))=.o)))
	lib_objects_$(1)    := $$(addprefix $$(OBJECT_$(UPPER_NAME)_DIR)/, $$(notdir $$(source_$(1):.$$(EXTENSION_$(UPPER_NAME))=.o)))
	objects_$(1)        := $$(target_objects_$(1)) $$(lib_objects_$(1))
	dependencies_$(1)   := $$(objects_$(1):.o=.d)
	targets_$(1)        := $$(addprefix $$(BINARY_DIR)/, $$(notdir $$(target_objects_$(1):.o=)))
endef

# Extract source, object code and executables. This also defines useful macros.
$(foreach impl, $(IMPLEMENTATIONS), $(eval $(call build_dirs,$(impl))))

# Build all implementations
all: $(IMPLEMENTATIONS)

# Compile with helpful warnings (-Wall -Wextra flags)
debug: CXXFLAGS  += $(WARNINGS) $(DEBUG)
debug: CUDAFLAGS += -g -G
debug: $(IMPLEMENTATIONS)

# Rule for creating directories
%/:
	@$(MKDIR) $@

# Rule for each implementation
.SECONDEXPANSION:
$(IMPLEMENTATIONS): LDFLAGS ?= -L $(OBJECT_$(UPPER_NAME)_DIR)
$(IMPLEMENTATIONS): $$(dirs_$$@) check-structure $$(targets_$$@)
	@$(PRINT) "$(ANSI_GREEN)Built $(call to_uppercase, $@) implementation successfully$(ANSI_RESET)"

# List the prerequisites for building your executable, and fill its
# recipe to tell make what to do with these
$(BINARY_DIR)/%: $(OBJECT_OMP_DIR)/%.o $(lib_objects_omp)
	$(LINK.cpp) $^ $(LOADLIBES) $(LDLIBS) -o $@

# Create CUDA binary
$(BINARY_DIR)/%: $(OBJECT_GPU_DIR)/%.o $(lib_objects_gpu)
	$(NVCC) $^ $(CUDAFLAGS) $(CULDLIBS) -o $@

# Since your source and object files don't share the same prefix, you
# need to tell make exactly what to do since its built-in rules don't
# cover your specific case
.SECONDEXPANSION:
$(OBJECT_OMP_DIR)/%.o: $(SOURCE_OMP)/%.cpp $$(wildcard $$(INCLUDE_OMP)/*.h)
	$(COMPILE.cpp) $(OUTPUT_OPTION) $<

.SECONDARY: $(objects_omp)

# Rules for compiling object code for CUDA implementation
.SECONDEXPANSION:
$(OBJECT_GPU_DIR)/%.o: $(SOURCE_GPU)/%.cu $$(wildcard $$(INCLUDE_GPU)/*.h)
	$(NVCC) $(CUXXFLAGS) $(CUDAFLAGS) $< -c -o $@

clean:
	$(RM) $(OBJECT_DIR) $(BINARY_DIR)

ifneq "$(MAKECMDGOALS)" "clean"
	-include $(dependencies_omp)
endif

# Warn if the repository structure does not match implementation targets
check-structure: INCLUDE_SUBTREE := $(call find_subdirs, $(INCLUDE_DIR))
check-structure: SOURCE_SUBTREE  := $(call find_subdirs, $(SOURCE_DIR))
check-structure:
	@if [ "$(IMPLEMENTATIONS)" != "$(INCLUDE_SUBTREE)" ] || [ "$(IMPLEMENTATIONS)" != "$(SOURCE_SUBTREE)" ]; then \
        $(PRINT) "$(ANSI_RED)Warning$(ANSI_RESET): repository tree does not match target implementations.";       \
        $(PRINT) "";                                                                                              \
        $(PRINT) "Expected directory structure:";                                                                 \
        $(PRINT) "$(ANSI_LIGHT_YELLOW)  polymps/";                                                                \
        $(PRINT) "    ├── $(SOURCE_DIR)/";                                                                        \
        count=0;                                                                                                  \
        for dir in $(IMPLEMENTATIONS); do                                                                         \
            count=$$((count+1));                                                                                  \
        done;                                                                                                     \
        current=0;                                                                                                \
        for dir in $(IMPLEMENTATIONS); do                                                                         \
            current=$$((current+1));                                                                              \
            if [ $$current -eq $$count ]; then                                                                    \
                $(PRINT) "    │   └─ $$dir/";                                                                     \
            else                                                                                                  \
                $(PRINT) "    │   ├─ $$dir/";                                                                     \
            fi;                                                                                                   \
        done;                                                                                                     \
        $(PRINT) "    │";                                                                                         \
        $(PRINT) "    ├── $(INCLUDE_DIR)/";                                                                       \
        current=0;                                                                                                \
        for dir in $(IMPLEMENTATIONS); do                                                                         \
            current=$$((current+1));                                                                              \
            if [ $$current -eq $$count ]; then                                                                    \
                $(PRINT) "    │   └─ $$dir/";                                                                     \
            else                                                                                                  \
                $(PRINT) "    │   ├─ $$dir/";                                                                     \
            fi;                                                                                                   \
        done;                                                                                                     \
        $(PRINT) "    │";                                                                                         \
        $(PRINT) "    └── ...$(ANSI_RESET)";                                                                      \
        $(PRINT) "Ensure that directories for each implementation (omp, gpu, etc) exist in both $(SOURCE_DIR)/ and $(INCLUDE_DIR)/."; \
    fi

help:
	@$(PRINT) 'Usage: make [options] [target] ...'
	@$(PRINT)
	@$(PRINT) 'Build targets for this project are:'
	@$(PRINT) '  all     - Compile and link all source files.'
	@$(PRINT) '  debug   - Build project with debugging options.'
	@$(PRINT) '  omp     - Build OpenMP implementation.'
	@$(PRINT) '  gpu     - Build CUDA implementation.'
	@$(PRINT) '  clean   - Remove all intermediate files.'
	@$(PRINT) '  help    - Display this information.'
	@$(PRINT)
	@$(PRINT) 'For more information on make options, refer to the Makefile documentation.'
	@$(PRINT) 'This Makefile is designed exclusively for UNIX systems.'

