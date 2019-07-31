# Bag-of-Pattern-Features (BOPF)

This repository contains the code accompanying the paper, "[Linear Time Complexity Time Series Classification with Bag-of-Pattern-Features](https://ieeexplore.ieee.org/abstract/document/8215500)" (Xiaosheng Li and Jessica Lin, ICDM 2017). This paper proposes a time series classification algorithm that has linear time complexity (training and testing).

## To Compile the Code

Assume using a Linux system:

`g++ -O3 -o BOPF BOPF.cpp -std=c++11`

## To Run the Code

`./BOPF [datasetname]`

\[datasetname\] is the name of the dataset to run, the user needs to place a folder named with the \[datasetname\] and the folder contains a training file datasetname_TRAIN and a testing file datasetname_TEST (The [UCR-Archive](https://www.cs.ucr.edu/~eamonn/time_series_data/) format). Please see the FaceFour example contained in the directory.

## Example

`./BOPF FaceFour`

Output:

```
The training time is 0.670000 seconds
The testing time is 0.150000 seconds
The testing error rate is 0
```

## Note

The code uses a char array buffer of size 100000 to read each line of the input file, so if the time series to use is very long, the characters that each line the input file contains may surpass the limit. In this case the buffer limit (line 31 of BOPF.cpp, MAX_PER_LINE) should be enlarged correspondingly.

## Citation
```
@inproceedings{li2017linear,
  title={Linear time complexity time series classification with bag-of-pattern-features},
  author={Li, Xiaosheng and Lin, Jessica},
  booktitle={2017 IEEE International Conference on Data Mining (ICDM)},
  pages={277--286},
  year={2017},
  organization={IEEE}
}
```
