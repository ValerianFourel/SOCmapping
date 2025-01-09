#!/bin/bash
find . -type f -size +30M | while read file; do
    # Remove ./ from the beginning of file path
    file="${file#./}"
    # Ignore if it's already in .gitignore
    if ! grep -q "^${file}$" .gitignore; then
        echo "$file" >> .gitignore
    fi
done
