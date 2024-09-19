#!/bin/bash

function show_help {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -c    Compress credentials"
    echo "  -e    Expand example files"
    echo "  -x    Extract compressed credentials"
    echo "  -h    Show this help message"
}

function compress_credentials {
    # Create a list of files to compress
    git ls-files "*.example" > example_files.txt

    # Remove the .example extension from each line and save to a new file
    sed 's/\.example$//' example_files.txt > credential_files.txt

    # Create tar archive
    tar -czf credentials.tar.gz -T credential_files.txt

    # Clean up temporary files
    rm example_files.txt credential_files.txt

    echo "Credentials compressed to credentials.tar.gz"
}

function expand_examples {
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
}

function extract_credentials {
    if [ ! -f "credentials.tar.gz" ]; then
        echo "Error: credentials.tar.gz not found"
        exit 1
    fi

    tar -xzf credentials.tar.gz
    echo "Credentials extracted from credentials.tar.gz"
}

# Check if no arguments were provided
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

# Parse command line options
while getopts "cexh" opt; do
    case ${opt} in
        c )
            compress_credentials
            ;;
        e )
            expand_examples
            ;;
        x )
            extract_credentials
            ;;
        h )
            show_help
            ;;
        \? )
            echo "Invalid option: $OPTARG" 1>&2
            show_help
            exit 1
            ;;
    esac
done
