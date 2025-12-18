## Overview
"Campus Dash" is a time-sensitive scavenger hunt game in which players, known as Dashers, compete to
fulfill tasks following a specific deadline. The tasks have specific locations and appear at random times with randomly generated reward values. This project implements a "smart brain" algorithm that batches the available tasks and dashers in a specific time interval, then, uses the Hungarian Algorithm to optimize the matching between tasks and users in a specific batch to maximize the total system reward. Additionally, it implements a time series forecasting pipeline for predicting rewards across each location to inform a better recommendation algorithm.

## Installation

Run the following command to use the package manager pip to install dependencies.

```bash
pip install -r 'requirements.txt'
```

## Usage Files
The baseline algorithm is implemented in simulator_baseline.py 
To run baseline simulator, run following command (at root directory):

```python
python simulator_baseline.py
#events processed: xyz
# Total reward: xyz
...
```
The the improved algorithm is implemented in simulator_improved.py
To run improved simulator, run following command (at root directory):

```python
python simulator_improved.py
#events processed: xyz
# Total reward: xyz
...
```

The code for both is already set that all that it will run when you run the code. However, if you want to make changes to the dasher or tasklog file, or the reward range, please make the appropriate changes in the __main__ program block. All relevant files for the simulator can be found in the folder project_files.

## 
