# TraceNet: An Effective Deep-Learning-Based Method for Baseline Correction of Near-Field Acceleration Records

## Early demo version
Demo code of TraceNet includes an example and demo data.

We recommend users run `example.py` and read the comments in that.


Thank you for using this early version of TraceNet.

More details and functions can be appended to TraceNet, if you have any problem, suggestion, or requirement, please contact me (dongsh@mail.ustc.edu.cn), or leave a message in issues.


## Relative Article
https://doi.org/10.1785/0220220272


## Copyright 

Xiaofei Chen Research Group
Department of Earth and Space Sciences, SUSTech, China.

## Requirements

This software package is based on python 3. 

We recommend using conda environment for usage.

### Python version

Python >= 3.5

### Dependent software package

- numpy
- obspy
- scipy
- matplotlib
- tensorflow>=2.4

## Usage

### import package

```
import tracenet
```


### functions

#### 1. tracenet.get\_file\_list(basis_dir="./", begin="", end="")

Obtain the file list from `basis_dir`, you can limit the file name's header and tile by changing the `begin` and `end` values.

The return value of this function is a string list of the selected file list.

#### 2. tracenet.load_model(modelFilePath)

Load TraceNet's network model file from the path `modelFilePath`.

The finish training model file is attached in this package as 'TraceNet_Finished.h5'.

Or you can use another model file. The training part of TraceNet will be released later.

This function returns a `tf.keras` object, which can be used in the baseline extraction function as follows.

#### 3. tracenet.extrace_baseline(model, testData, shift=10)

Extract baseline drifts using the given model object `model` from `testData`.

`testData` should be a numpy array with 1-D data trace. 

`shift` value is the boundary cutoff pixel width, which is applied to reduce performance passivation in the network boundary.

By using this value, the network will abandon the given length of sampling data point in the header and tile in network prediction. 

The input trace `testData` will be interpolated and merged with the shift part. Thus the length of return data is constant.

This function returns the baseline extracted from the input data trace. The length of the return baseline array is the same as the input trace.


#### 4. plot_nez(accNEZ, veloNEZ, baselineNEZ, dispNEZ, offsetNEZ, dt, fileName, manualN=[], manualE=[], manualZ=[])

A plot function for quickly inspecting results of correction.

The parameters are:

- accNEZ: 3 channel accelerograms in NS-EW-UD order.
- veloNEZ: 3 channel velocity traces in NS-EW-UD order.
- baselineNEZ: 3 channel extracted baselines in NS-EW-UD order.
- dispNEZ: 3 channel corrected displacements in NS-EW-UD order.
- offsetNEZ: 3 channel permanent ground offset in NS-EW-UD order.
- dt: delta value of data, for plotting the correct time axis.
- fileName: save file path.

You can check `example.py` for details of this function usage.







