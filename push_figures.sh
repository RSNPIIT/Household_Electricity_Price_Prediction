#!/bin/bash

# Loop through all untracked PNG files in the figures directory
for file in figures/*.png; do
    # Check if the file actually exists to avoid loop errors
    if [ -f "$file" ]; then
        echo "----------------------------------------"
        echo "Processing: $file"
        
        # Extract the base filename for a clean commit message (e.g., fig02_power_consumption_time_series.png)
        filename=$(basename "$file")
        
        # Stage the single file
        git add "$file"
        
        # Commit with a clear message
        git commit -m "Add $filename"
        
        # Push to the remote repository
        git push origin main
    fi
done

echo "----------------------------------------"
echo "All figures have been pushed individually!"

