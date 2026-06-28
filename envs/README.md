# Package environment setup

Different installation strategies are provided for different use cases. 
To minimize dependency requirements, 
we strongly recommend selecting the installation option that 
best matches your intended workflow and available resources, 
including whether the required components have already been obtained from the authors.

Before installing any environment, please make sure that the required code repository can be downloaded locally.
```shell
git clone https://github.com/BGI-SynBio/SPIDER-WEB.git 
```

Please note that, except for the "basic setup", all other setups require a Linux environment.

## Basic setup

If you intend to use the basic Python implementation of SPIDER-WEB 
for standard read/write workflows or evaluation tasks, 
please install the required environment using the following instructions:

```shell
cd SPIDER-WEB/envs
conda env create -f spiderweb-base.yml
```

Before running any related programs, activate the installed environment:

```shell
conda activate spiderweb-base
```

## Full setup

If you intend to use the full implementation of SPIDER-WEB, 
including both Python and C++ components, 
for standard read/write workflows or evaluation tasks, 
please install the required environment using the following instructions:

```shell
cd SPIDER-WEB/envs
conda env create -f spiderweb-full.yml
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

Please note that the executable file for the C++ version, `accelerator`, 
should be obtained by contacting the corresponding authors. 
After placing it under the [`sbw`](https://github.com/BGI-SynBio/SPIDER-WEB/tree/main/sbw) directory, 
grant it executable permission.

```shell
cd ../sbw
chmod +x accelerator
```

To verify that the setup has been completed correctly, enter:

```shell
./accelerator
```
The expected output is:

```terminaloutput
Usage:          ./accelerator  <command> [options]
Command list:
        repair              repair the DNA sequence using accelerator
        decode              decode the DNA sequence to the original binary sequence
        --help              show this message
```

Before running any related programs, activate the installed environment:

```shell
conda activate spiderweb-full
```

## Replicability-oriented setup

If you intend to reproduce all experiments reported in the SPIDER-WEB publication, 
including all steps described under the [`works`](https://github.com/BGI-SynBio/SPIDER-WEB/tree/main/works) directory,
please install the required environment using the following instructions:

```shell
cd SPIDER-WEB/envs
conda env create -f spiderweb-repl.yml
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

Please note that the executable file for the C++ version, `accelerator`, 
should be obtained by contacting the corresponding authors. 
After placing it under the [`sbw`](https://github.com/BGI-SynBio/SPIDER-WEB/tree/main/sbw) directory, 
grant it executable permission.

```shell
cd ../sbw
chmod +x accelerator
```

To verify that the setup has been completed correctly, enter:

```shell
./accelerator
```
The expected output is:

```terminaloutput
Usage:          ./accelerator  <command> [options]
Command list:
        repair              repair the DNA sequence using accelerator
        decode              decode the DNA sequence to the original binary sequence
        --help              show this message
```

Before running any related programs, activate the installed environment:

```shell
conda activate spiderweb-repl
```

Since this environment includes third-party bioinformatics tools, 
we recommend checking their availability before conducting evaluation or comparison experiments.
