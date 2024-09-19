#!/bin/bash

# Get all .example files
example_files=$(git ls-files "*.example")

# Loop through each file
for file in $example_files; do
    # Remove the .example extension
    new_file="${file%.example}"
    
    # Check if the new file already exists
    if [ -f "$new_file" ]; then
        echo "Skipping $new_file: File already exists"
    else
        # Copy the content of the .example file to the new file
        cp "$file" "$new_file"
        echo "Created $new_file from $file"
    fi
done