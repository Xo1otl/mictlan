#!/bin/bash

# Create a list of files to compress
git ls-files "*.example" > example_files.txt

# Remove the .example extension from each line and save to a new file
sed 's/\.example$//' example_files.txt > credential_files.txt

# Create tar archive
tar -czf credentials.tar.gz -T credential_files.txt

# Clean up temporary files
rm example_files.txt credential_files.txt

echo "Credentials compressed to credentials.tar.gz"