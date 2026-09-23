<p>
<img src="./logo.svg" alt="SPIDER-WEB" title="SPIDER-WEB" width="100%"/>
</p>

# SPIDER-WEB enables a real-time data retrieval of DNA-based data storage

[![CircleCI](https://circleci.com/gh/BGI-SynBio/SPIDER-WEB/tree/main.svg?style=shield)](https://circleci.com/gh/BGI-SynBio/SPIDER-WEB/tree/main)
[![PythonVersion](https://img.shields.io/badge/python-3.10-blue)](https://img.shields.io/badge/python-3.10-blue)
[![Licence](https://img.shields.io/badge/Licence-BGI_Research_v1-orange.svg)](https://github.com/BGI-SynBio/SPIDER-WEB/blob/main/LICENSE.pdf)
[![Publication](https://img.shields.io/badge/Publication-Nature_Computational_Science-red.svg)]()

Data retrieval efficiency remains one of the key challenges 
hindering the practical application of DNA-based data storage, 
being at least five orders of magnitude lower than that of compact discs. 
Here, we introduce SPIDER-WEB, an all-in-one coding framework that eliminates reliance on 
time-consuming sequence clustering and alignment steps in conventional pipelines, 
allowing data processing to proceed concurrently with the sequencing process. 
Across 3,000 synthesised data-coding DNA sequences, 
SPIDER-WEB demonstrated an efficiency improvement of 587-fold to 9,083-fold over optimised conventional pipelines. 
Projections for megabyte and exabyte-scale retrievals show even greater gains, 
with efficiency enhancements of 3.0-3.2 and 24.0-29.7 orders of magnitude, respectively. 
On a laptop, SPIDER-WEB achieves retrieval throughput up to 13.8 MB/s. 
When integrated with nanopore sequencing, 
SPIDER-WEB enabled progressive rendering of a 19,200-byte video during sequencing, 
with full retrieval completed within 100 seconds. 
Overall, we demonstrate that the compact-disc-level retrieval performance is achievable in DNA-based data storage.

## Repository structure

The core codebase of this repository consists primarily of three directories: 
- The [`sbw`](https://github.com/BGI-SynBio/SPIDER-WEB/tree/main/sbw) directory contains the source code of SPIDER-WEB. 
- The [`tests`](https://github.com/BGI-SynBio/SPIDER-WEB/tree/main/tests) directory contains automated tests covering the major operations of the source code. 
- The [`works`](https://github.com/BGI-SynBio/SPIDER-WEB/tree/main/works) directory contains the full execution workflows for all simulations and experiments presented in the manuscript.

In addition to these core directories, 
this code repository contains several auxiliary directories that 
support the use, testing, and maintenance of the core codebase. 
That is, the [`envs`](https://github.com/BGI-SynBio/SPIDER-WEB/tree/main/envs) directory
provides dependency environments tailored to different use cases,
and the [`.circleci`](https://github.com/BGI-SynBio/SPIDER-WEB/tree/main/.circleci) directory
contains the CircleCI configuration file used to run the required repository-level self-checks.

## Installation

Please make sure that this code repository has already been downloaded locally. 
Otherwise, you need to run the following command:

```shell
git clone https://github.com/BGI-SynBio/SPIDER-WEB.git 
```

To minimise dependency requirements, 
we strongly recommend selecting the environment that 
best matches your intended workflow and available resources, 
including whether the required components have already been obtained from the authors.
Please refer to the provided guidelines 
in the [`envs`](https://github.com/BGI-SynBio/SPIDER-WEB/tree/main/envs) directory
to install the appropriate environment according to your intended use.

## Replicability

To support the development of the relevant community and 
to provide a practical reference for research-software disclosure and related replicability initiatives, 
this code repository provides the materials needed 
to replicate the computational procedures underlying the publication.

In this repository, replicability refers to the complete process of 
generating the figures and tables reported in the publication 
from their corresponding starting points. 
Depending on the workflow, users can start either from the experimental data or 
from predefined algorithmic procedures that generate the required outputs during execution.
By following the procedures provided in 
the [`works`](https://github.com/BGI-SynBio/SPIDER-WEB/tree/main/works) directory, 
users can regenerate intermediate raw outputs, 
convert these outputs into packaged data files, 
and use the packaged data files directly for plotting the figures and tables included in the publication.
For more information, 
please refer to the [`README.md`](https://github.com/BGI-SynBio/SPIDER-WEB/blob/main/works/README.md) file 
in the [`works`](https://github.com/BGI-SynBio/SPIDER-WEB/tree/main/works) directory.

Please note that the support provided through this code repository 
is primarily intended for replicating the computational procedures and results reported in the publication. 
We do not undertake to support unrelated studies or directly competitive follow-up studies.
Unless otherwise agreed in writing, any study supported through this code repository 
is subject to the terms and conditions of the 
[`LINCECE.pdf`](https://github.com/BGI-SynBio/SPIDER-WEB/blob/main/LICENSE.pdf).

Some executable files used for acceleration purposes or hardware-coupled execution 
are not directly included in this code repository.
These files may be made available upon reasonable request to the 
[authors](https://github.com/BGI-SynBio/SPIDER-WEB/blob/main/CORRESPONDENCE.md).
for contact information and the information that should be included in such requests.
At this stage, experiments that require coordination with specific hardware devices 
cannot be directly replicated by third parties using this code repository alone. 
Requests related to the replicability of these hardware-coupled experiments,
should be discussed with the [authors](https://github.com/BGI-SynBio/SPIDER-WEB/blob/main/CORRESPONDENCE.md).

To further support transparent inspection of the replicability scope, 
this repository also provides a machine-readable evidence chain describing the relationships 
between the data elements and software components. 
This evidence chain is intended to help users identify which repository components are associated with 
each reported result and to inspect how computational procedures connect starting materials to final reported outputs. 

[//]: # (For more information, please refer to )

[//]: # ([`evidence.yml`]&#40;https://github.com/BGI-SynBio/SPIDER-WEB/blob/main/evidence.yml&#41;.)

## Artificial intelligence use restriction

No license is granted to use any Repository Content, 
in whole or in part, for the purpose of developing, training, fine-tuning, validating, testing, benchmarking, 
evaluating, improving, or otherwise enabling any artificial intelligence system, 
machine learning system, large language model, foundation model, generative model, 
code generation system, embedding model, retrieval-augmented generation system, or related automated model or system.

This restriction includes, without limitation, the use of Repository Content for dataset aggregation, 
corpus construction, embedding generation, synthetic data generation, model pre-training, model fine-tuning, 
model evaluation, benchmark construction, retrieval corpus construction, 
or any other process in which Repository Content is copied, extracted, transformed, encoded, indexed, 
or incorporated into a machine learning or artificial intelligence workflow.
Any such use requires prior express written permission from the copyright holders.
Common AI models are mentioned in [`robots.txt`](https://github.com/BGI-SynBio/SPIDER-WEB/blob/main/robots.txt).

For clarity, this restriction does not prohibit human reading, manual review, scholarly citation, 
reproducibility evaluation, or other uses expressly permitted under this Licence, 
provided that such uses do not involve the incorporation of Repository Content into 
an artificial intelligence or machine learning development workflow.

## Development history and priority records

SPIDER-WEB was developed over a period of approximately seven years, 
during which we accumulated substantial design records, empirical analyses, implementation experience, 
and numerous unsuccessful attempts related to this work. 
The corresponding publication reports only a small subset of this broader development history. 
Even within this condensed presentation, the corresponding Supplementary Information contains 15 appendices 
covering different technical aspects of the work.

Furthermore, in addition to this code repository, 
the development history of SPIDER-WEB is supported by 
an earlier [open-source repository](https://github.com/HaolingZHANG/DNASpiderWeb) 
containing 183 commits from Jul 28, 2021 to Jun 21, 2024, 
and by a more recent closed-source repository 
containing 143 commits from Aug 21, 2024 to Jun 28, 2026.
Researchers who are interested in the broader development history of this work may contact 
the [authors](https://github.com/BGI-SynBio/SPIDER-WEB/blob/main/CORRESPONDENCE.md). 

These records may also be relevant to legitimate intellectual-property and priority-protection considerations. 
Such materials may include records relevant to priority, inventorship, ownership, or related rights. 
Where necessary and under appropriate conditions, 
we are willing to provide relevant empirical records to substantiate the development history of this work 
and to support legitimate intellectual-property and priority-protection interests.

## Citation

If this repository is helpful to your research, or if you use this repository in your work, 
please consider citing the corresponding publication.
Here is the BibTeX entry:

````
@article{zhang2026spider,
  title={SPIDER-WEB enables a real-time data retrieval of DNA-based data storage},
  author={Zhang, Haoling and Lin, Xiaoyi and Lan, Zhaojun and Xu, Demin and Wang, Wen and Wang, Yun and Han, Tao and Deng, Ziqing and Wang, Yadong and Tegn\'{e}r, Jesper and Xu, Xun and Ping, Zhi and Zhang, Yiwei and Liu, Bo and Shen, Yue},
  journal={Nature Computational Science},
  volume={X},
  number={X},
  pages={XXX--XXX},
  year={2026},
  publisher={Nature Publishing Group}
}
````
