#!/bin/bash

# Configuration: Adjust these to your exact Region-Of-Interest (ROI)
WIDTH=98
HEIGHT=98
X_OFFSET=363   # Replace with your actual X offset from the top-left
Y_OFFSET=221   # Replace with your actual Y offset from the top-left

# Create output directory if it doesn't exist
OUTPUT_DIR="cropped"
mkdir -p "$OUTPUT_DIR"

echo "Starting batch crop (ROI: ${WIDTH}x${X_OFFSET}+${X_OFFSET}+${Y_OFFSET})..."

# Loop through all jpg files in the current directory
for img in *.jpg; do
    # Skip if no match found (prevents literal '*.jpg' execution)
    [ -e "$img" ] || continue
    
    echo "Processing: $img"
    
    # ImageMagick crop syntax: WxH+X+X
    # +repage resets the virtual canvas offset to keep standard clean headers
    convert-im6.q16 "$img" -crop "${WIDTH}x${HEIGHT}+${X_OFFSET}+${Y_OFFSET}" +repage "$OUTPUT_DIR/$img"
done

echo "Done! Cropped images saved to '$OUTPUT_DIR/'."
