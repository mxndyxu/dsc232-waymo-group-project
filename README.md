# DSC232 Waymo Group Project

Kristen Oleson<br>
Cory Ornelas<br>
Audrius Pasvenskas<br>
Mandy Xu<br>

## Dataset Overview

Dataset: [Waymo Open Motion Dataset](https://waymo.com/open/data/motion/)<br>
Number of observations: 832,346

Each observation corresponds to a single tracked vehicle trajectory extracted from raw scenario protobuf files after filtering and preprocessing.

## Data

**scenario_id (string, categorical)**<br>
A unique identifier for a driving scenario (scene). Each scenario contains multiple agents (vehicles) and represents a short driving clip.<br>
Scale: Nominal -- identifier, no numerical meaning<br>
Distribution: After aggregating by scenario_id, the scenario-level statistics are shown below.

Number of scenarios: 29,411<br>
Mean: 28.30 vehicles per scenario<br>
Std dev: 19.83<br>
Min: 1<br>
Max: 218<br>
Quantiles: [1, 14, 24, 37, 218]<br>

This shows a moderately right-skewed distribution where most scenarios contain a few dozen vehicles, but some dense traffic scenes contain significantly more.

**track_id (long, categorical)**<br>
A unique identifier for a specific vehicle within a scenario.<br>
Scale: Nominal -- identifier<br>
Distribution: ***TODO***

**past_x (array<double>, continuous), past_y (array<double>, continuous)**<br>
Sequences of x and y coordinates representing the observed past motion of a vehicle.

past_x[i], past_y[i] = position of the vehicle at timestep i in the past<br>
History length: 11 timesteps (~1 second of motion at 10 Hz)<br>
Scale: Continuous, ratio -- real-valued coordinates in meters in a local coordinate frame<br>
Distribution: ***TODO***

**future_x (array<double>, continuous), future_y (array<double>, continuous)**<br>
This is the target variable. Sequences of x and y coordinates representing the ground-truth future motion of the vehicle.

future_x[i], future_y[i] = position of the vehicle at timestep i in the future<br>
Prediction horizon: 80 timesteps (~8 seconds at 10 Hz)<br>
Scale: Continuous, ratio -- meters in the same coordinate frame as past trajectories<br>
Distribution: ***TODO***

## Missing and Duplicate Values

Missing values: None (invalid or incomplete trajectories are filtered out during preprocessing)<br>
Duplicate values: None detected in the final dataset
