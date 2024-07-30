#!/bin/bash

# File containing the list of modules
MODULES_FILE="modules.txt"

# Check if the file exists
if [[ -f "$MODULES_FILE" ]]; then
  # Read each line in the file and load the module
  while IFS= read -r module; do
    module load "$module"
  done < "$MODULES_FILE"
else
  echo "Modules file not found: $MODULES_FILE"
fi
