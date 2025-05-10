After building the library, you can run the code blocks in the [README.md](../README.md) by

```bash
cd sample_c
python extract_codeblocks.py # extract code blocks from sample_c.md and save to test_files/*.c
# set the path to the libsparseir installation
export SparseIR_DIR=$HOME/opt/libsparseir/share/cmake
cmake .
make run_all_tests
```
