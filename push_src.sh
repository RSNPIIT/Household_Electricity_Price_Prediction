#!/bin/bash

# Loop through all the restored files inside the src folder
for file in src/*.py; do
    if [ -f "$file" ]; then
        echo "----------------------------------------"
        echo "Restoring and pushing: $file"
        
        # Get the clean filename (e.g., eda.py)
        filename=$(basename "$file")
        
        # Stage the single file
        git add "$file"
        
        # Commit with an individual message
        git commit -m "Implement and optimize $filename core modules"
        
        # Push it immediately
        git push origin main
    fi
done

echo "----------------------------------------"
echo "All files have been restored and committed individually!"

