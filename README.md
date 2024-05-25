# Approach

The following steps, assumptions and abstractions were made to reach the final results.

## 1. PID Control 
The proposed abstracted system dynamics are well fit for a PI(D) controller. A PID controller was added to the controllers under pid.py. The controller was initially tuned manually for exploratory purposes. The PID controller was tested with the simulator and the results were satisfactory. The PID controller was then used as a baseline for the autotuning process.

## 2. Multi-generational Gradient Descent
The goal here is to reduce the average total cost (increase fitness), over each generation by means of genetic selection. Due to the amount of data available, we should be able to write a robust controller that is performant in as many enviromnents as possible. Some initial testing was done and the assumption was made that the multi-state dataset had varying types of results based on the data of these states. This means that tuning the controller on e.g. a 100 roads would underrepresent the controller on the rest of the dataset. For this reason an efficient and performant method had to be picked for the tuning process. 

A multi-generational tuning process was picked to slowly increase the sample size on the dataset in order to locate the local optima. In this case we scaled up in the following manner - 5, 50, 1000, 20.000, each decreasing the amount of initial runs on the potential PID space with the best performant PID values making it to the next generation. The amount of runs per generation were decreased in the following manner - 2000, 200, 20, 1. The final run on the whole sample set was conducted to determine the results on the dataset as a whole of the most performant PID controller. The initial generation was used as a warm start for the gradient descent algorithm, meaning that all subsequent generations searched neighboring nodes for more optimal values up to 10 epochs.

## 3. Optimisations 
A multitude of methods were used to optimise the above processes for the limited amount of compute resources available. Initially ThreadPoolExecutor was implemented, but due to the nature of the tinyphysics simulation script not all available GPU and CPU resources were used optimally. Mainly due to the fact that in the smaller generations the tuner is CPU bound. For this reaso ProcessPoolExecutor was piced to optimally use all resources available. Populations inside are generations are only loaded in as previous populations finish their descent to optimise for memory use. 

## 4. Hardware
Initial testing and tuning was done on an NVidia 2070. The final results were achieved on a cloud hosted NVidia A100 with 40GB of memory. Getting the corresponding drivers installed for the A100 was a fun task. I know there's pre-existing computation platforms out there, but I personally always enjoy a more bare bones approach directly from within a VM environment when the time allows me to. 


# Results

## 1. The tuning process
The tuning resulted in a PI controller with corresponding values 0.01846154 and 0.09230769 on the final sample 20k size. The average cost on this sample size was equal to 34.86. The effects of the derivative gain seem to be disproportial small to the other two parameters, which allows us to continue with a PI controller for the final benchmark.

## 2. Final benchmark
The tuned PID controller surpasses the provided baseline on the given benchmarks. The final results for the provided benchmark can be found under [results.html](link).


# Recommendations

## 1. Multi-GPU
Use something like pytorch to extend the runtime to distribute the training of the genetic algorithm across multiple GPUs. 

## 2. Max error study
Study max error of controller across all independent roads to determine fault tolerance. Currently only the average values of the tuned PID are used for performance benchmarking. Outliers should be taken into account and studied to determine tolerance against edge environments.

## 3. Adaptive PI(D)
Make PI(D) adaptive to new environments and vehicles, within bounds, to enable it to keep learning tuning on a user's direct environment. Most of the time users have a set amount of routes they take on a daily basis. Training of the "personal" PID, say on a rolling basis for the past 300 drives, could be done online using cloud computing. Or offline, using a GPU heavy machine at home (wink wink)... maybe a tinybox. Due to the open source nature of Comma, this could be turned into an on/off functionality, and could potentially be even considered as an expansion of service (online) / reason to get a tinybox (offline).


