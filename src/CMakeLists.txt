cmake_minimum_required(VERSION 3.16)
project(MSLTransformer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required frameworks
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Source files for the Transformer model
set(TRANSFORMER_SOURCES
    host/transformer_model.mm
    host/transformer_model_impl.mm
)

# Create the transformer model library
add_library(transformer_model ${TRANSFORMER_SOURCES})
target_link_libraries(transformer_model ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})

# Test executable
add_executable(test_transformer_model
    ../tests/test_transformer_model.mm
)
target_link_libraries(test_transformer_model transformer_model ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})

# Training test executable
add_executable(test_transformer_training
    ../tests/test_transformer_training.mm
)
target_link_libraries(test_transformer_training transformer_model ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})

# Add compiler flags for Objective-C++
target_compile_options(transformer_model PRIVATE -x objective-c++)
target_compile_options(test_transformer_model PRIVATE -x objective-c++)
target_compile_options(test_transformer_training PRIVATE -x objective-c++)

# Set the correct Metal framework path
if(APPLE)
    set_target_properties(transformer_model PROPERTIES
        COMPILE_FLAGS "-framework Metal -framework Foundation"
        LINK_FLAGS "-framework Metal -framework Foundation"
    )
    set_target_properties(test_transformer_model PROPERTIES
        COMPILE_FLAGS "-framework Metal -framework Foundation"
        LINK_FLAGS "-framework Metal -framework Foundation"
    )
    set_target_properties(test_transformer_training PROPERTIES
        COMPILE_FLAGS "-framework Metal -framework Foundation"
        LINK_FLAGS "-framework Metal -framework Foundation"
    )
endif()

# Optional: Add a simple example executable
add_executable(transformer_example
    examples/simple_transformer.mm
)
target_link_libraries(transformer_example transformer_model ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
target_compile_options(transformer_example PRIVATE -x objective-c++)

if(APPLE)
    set_target_properties(transformer_example PROPERTIES
        COMPILE_FLAGS "-framework Metal -framework Foundation"
        LINK_FLAGS "-framework Metal -framework Foundation"
    )
endif() 