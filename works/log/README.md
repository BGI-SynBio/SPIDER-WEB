# Execution log folder

This folder stores the execution logs generated during the execution of the following pipeline scripts:

- [`step_1_preparation.py`](https://github.com/BGI-SynBio/SPIDER-WEB/tree/main/works/step_1_preparation.py)
- [`step_2_simulations.py`](https://github.com/BGI-SynBio/SPIDER-WEB/tree/main/works/step_2_simulations.py)
- [`step_3_experiments.py`](https://github.com/BGI-SynBio/SPIDER-WEB/tree/main/works/step_3_experiments.py)
- [`step_4_supplements.py`](https://github.com/BGI-SynBio/SPIDER-WEB/tree/main/works/step_4_supplements.py)

Each `.txt` file contains a subset of the execution logs produced by a specific script. 
When a monitoring step is provided during execution, 
the corresponding log file matches the monitored section in the code. 
In this sense, the log files provide a direct record of the execution status for the corresponding pipeline components.

These records are provided for two main purposes. 
First, they document that the relevant workflows were executed according to the predefined procedures. 
Second, they provide users who wish to repeat the workflows with direct reference expectations for 
the running process and final execution status.

The error-tolerant experiments used for Fig. 3 and Fig. 4, 
as well as the efficiency-approximation simulations used for Fig. 5, 
were conducted on a primary test system running Ubuntu 24.04.1 LTS, 
equipped with a 13th Gen Intel® Core™ i5-13400 CPU and 32 GB DDR4 RAM. 
The reported runtime-related results are based exclusively on this platform.
For improved visualisation, 
the demonstration experiments presented in Fig. 6 were conducted on a separate system running Windows 11 24H2, 
equipped with a 14th Gen Intel® Core™ i9-14900HX CPU and 32 GB DDR5 RAM.
All experiments and simulations that were not used for runtime-sensitive reporting 
were conducted across multiple computational environments, 
including personal workstations, the KAUST IBEX high-performance cluster, and the BGI cloud infrastructure.