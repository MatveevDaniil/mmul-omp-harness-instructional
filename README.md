# Matrix Multiplication parallelization study

To repeat our experiments on Perlmutter, you firstly need to download and compile the source code:
```
> git clone https://github.com/MatveevDaniil/mmul-omp-harness-instructional.git
> cd mmul-omp-harness-instructional
> mkdir build; cd build; cmake ..; make; cd ..
```

Then allocate the CPU node, run the script of experiments and script for visualization:
```
> salloc --nodes 1 --qos interactive --time 00:20:00 --constraint cpu --account=m3930
> bash run_tests
> python parse_result.py
```