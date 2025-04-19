#rm -rf ./build && cmake -S . -B ./build -DSPARSEIR_BUILD_TESTING=ON && cmake --build ./build -j && ./build/test/libsparseirtests
cmake -S . -B ./build -DSPARSEIR_BUILD_TESTING=ON && cmake --build ./build -j && ./build/test/libsparseirtests
