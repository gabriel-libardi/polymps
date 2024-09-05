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
CPPFLAGS ?=  -Ieigen -Ijson/single_include/nlohmann -Ilibigl/include

# Link OpenMP code to stdc++fs
LDLIBS  := -lstdc++fs

# CUDA compiler definitions
NVCC         := nvcc
CUDAFLAGS    ?= -I/mnt/nfs/modules/apps/cuda-toolkit/9.0.176/samples/common/inc
CUXXFLAGS    := -std=c++11 
CULDLIBS     :=
CULDFLAGS    :=
CUDADEBUG    := -g -G

# Empty variable, equivalent to NULL in C
EMPTY        :=

# Maps Systematic names of compilation config variables to their usual names.
CC_OMP            := COMPILE.cpp
CC_GPU            := NVCC

# Compilation Macros for each implementation
LK_OMP            := LINK.cpp
LK_GPU            := NVCC

# Pre-compilation flags for nvcc and g++
CC_FLAGS_OMP      := EMPTY
CC_FLAGS_GPU      := CUDAFLAGS

CCXX_FLAGS_OMP    := EMPTY
CCXX_FLAGS_GPU    := CUXXFLAGS

# Linker flags macros for OMP and CUDA implementation
LDFLAGS_OMP       := LOADLIBES
LDFLAGS_GPU       := CULDFLAGS

LDLIBS_OMP        := LDLIBS
LDLIBS_GPU        := CULDLIBS

INCLUDE_PATHS_OMP := CPPFLAGS
INCLUDE_PATHS_GPU := CUDAFLAGS

# Directives that are not filenames to be built
.PHONY: all debug $(IMPLEMENTATIONS) clean help check-structure

# Define useful Macros and routines
# This routine returns the subdirectories of a given PATH
define find_subdirs
$(shell find $(1) -maxdepth 1 -type d -exec basename {} \; | grep -v $(1))
endef

# This routine returns a given string in all caps.
define to_uppercase
$(shell $(PRINT) $(1) | tr '[:lower:]' '[:upper:]')
endef

# Get 
define get_option
$(shell basename $(1) | tr '[:lower:]' '[:upper:]')
endef

# Define ANSI color codes
ANSI_RED          := \x1b[31m
ANSI_GREEN        := \x1b[32m
ANSI_LIGHT_YELLOW := \x1b[33m
ANSI_RESET        := \x1b[0m

# Useful commands definitions
MKDIR := mkdir -p
RM    := rm -rf
PRINT := echo -e

# Implementations are different versions of PolyMPS made for different computational demands
# An OpenMP and a CUDA cimplementation are available currently
IMPLEMENTATIONS := $(call find_subdirs, $(SOURCE_DIR))
UPPER_IMPL      := $(call to_uppercase, $(IMPLEMENTATIONS))

# Creates build directories if they do not exist
define build_dirs
	$(eval UPPER_NAME              := $(call to_uppercase,$(1)))
    $(eval $(1)                    := $(UPPER_NAME))

	$(eval OBJECT_$(UPPER_NAME)_DIR       :=   $(OBJECT_DIR)/$(1))
	$(eval INCLUDE_$(UPPER_NAME)          :=   $(INCLUDE_DIR)/$(1))
	$(eval SOURCE_$(UPPER_NAME)           :=   $(SOURCE_DIR)/$(1))
    $(eval BINARY_$(UPPER_NAME)_DIR       :=   $(BINARY_DIR)/$(1))
    $(eval $(INCLUDE_PATHS_$(UPPER_NAME)) += -I$(INCLUDE_DIR)/$(1))

	$(eval source_$(1)           := $$(wildcard $$(SOURCE_$(UPPER_NAME))/*.$$(EXTENSION_$(UPPER_NAME))))
	$(eval target_objects_$(1)   := $$(addprefix $$(OBJECT_$(UPPER_NAME)_DIR)/, $$(notdir $$(TARGET_$(UPPER_NAME):.$$(EXTENSION_$(UPPER_NAME))=.o))))
	$(eval LIB_$(UPPER_NAME)     := $$(addprefix $$(OBJECT_$(UPPER_NAME)_DIR)/, $$(notdir $$(source_$(1):.$$(EXTENSION_$(UPPER_NAME))=.o))))
	$(eval dependencies_$(1)     := $$(objects_$(1):.o=.d))
	$(eval TARGETS_$(UPPER_NAME) := $$(addprefix $$(BINARY_$(UPPER_NAME)_DIR)/, $$(notdir $$(target_objects_$(1):.o=))))
endef

# Extract source, object code and executables. This also defines useful macros.
$(foreach impl, $(IMPLEMENTATIONS), $(eval $(call build_dirs,$(impl))))

# OpenMP configs
.SECONDARY: $(OBJECTS_OMP)
LDFLAGS     ?= -L $(OBJECT_OMP_DIR)

# Build all implementations
all: $(IMPLEMENTATIONS)

# Compile with helpful warnings (-Wall -Wextra flags)
debug: CXXFLAGS  += $(WARNINGS) $(DEBUG)
debug: CUDAFLAGS += $(CUDADEBUG)
debug: $(IMPLEMENTATIONS)

# Rule for creating directories
%/:
	@$(MKDIR) $@

# Rule for each implementation
.SECONDEXPANSION:
$(IMPLEMENTATIONS): check-structure $$(TARGETS_$$($$@)) 
	@$(PRINT) "$(ANSI_GREEN)Built $($@) implementation successfully$(ANSI_RESET)"

# Link objects into binary for each implementation
$(BINARY_DIR)/%: $$(OBJECT_DIR)/%.o $$(LIB_$$($$(*D))) $$(@D)/
	$($(LK_$($(*D)))) $(LIB_$($(*D))) $($(LDFLAGS_$($(*D)))) $($(LDLIBS_$($(*D)))) -o $@

# Compile object code for each implementation
$(OBJECT_DIR)/%.o: $(SOURCE_DIR)/%.* $$(@D)/
	$($(CC_$($(*D)))) $($(CC_FLAGS_$($(*D)))) $($(CCXX_FLAGS_$($(*D)))) $< -o $@

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

