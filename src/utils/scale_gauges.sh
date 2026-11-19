#!/bin/bash

# Configuration: Adjust these to your exact Region-Of-Interest (ROI)
WIDTH=32
HEIGHT=32

# Create output directory if it doesn't exist
OUTPUT_DIR="scaled"
mkdir -p "$OUTPUT_DIR"

echo "Starting batch scale to ${WIDTH}x${X_OFFSET}..."

# Loop through all jpg files in the current directory
for img in *.jpg; do
    # Skip if no match found (prevents literal '*.jpg' execution)
    [ -e "$img" ] || continue
    
    echo "Processing: $img"
    
    # ImageMagick scale syntax: WxH+X+X
    # +repage resets the virtual canvas offset to keep standard clean headers
    convert-im6.q16 "$img" -resize "${WIDTH}x${HEIGHT}" +repage "$OUTPUT_DIR/$img"
done

echo "Done! Scaled images saved to '$OUTPUT_DIR/'."
