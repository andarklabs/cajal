#!/bin/bash

# Set the project directory
project_dir="/Users/andrewceniccola/Desktop/cajal/src/"

# Compilation command with c++11, no version warnings, -I option and access to our innate files and gives access to mat_math.cpp and throws errors to terminal
cd "$project_dir" && g++ -std=c++11 -Wc++11-extensions -I./math initializers/initializers.cpp math/mat_math.cpp -o initializers/executive 2>&1

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Running the program..."

    # Run the compiled program
    "$project_dir/initializers/executive"

    # Delete the compiled file
    rm "$project_dir/initializers/executive"
else
    echo "Compilation failed."
fi

# remember chmod +x initializers/initializers.sh to give permissions
# run from src as ./initializers/initializers.sh
