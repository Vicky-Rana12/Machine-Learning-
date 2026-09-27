#!/bin/bash
for file in Experiment*.py; do
    num=$(echo "$file" | grep -oE '[0-9]+' | head -1)
    if [ -n "$num" ]; then
        folder="Experiment-$num"
        mkdir -p "$folder"
        git mv "$file" "$folder/"
        echo "Moved $file -> $folder/"
    fi
done
