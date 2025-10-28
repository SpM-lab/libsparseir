export SparseIR_DIR=$HOME/opt/libsparseir/share/cmake
cmake -S . -B ./build
cmake --build build --target test
