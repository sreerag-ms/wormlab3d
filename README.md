# Wormlab3D

> **Credits**: This project is forked from the original [wormlab3d](https://github.com/UoL-wormlab/wormlab3d) repository by The Leeds Wormlab.

## Installation

### Prerequisites

- Python 3.12+
- Conda or Miniconda
- This project requires wormlab3d database

### Setup Environment

Clone the repository:
```bash
git clone https://github.com/sreerag-ms/wormlab3d.git
cd wormlab3d
```

Create a Python 3.9 conda environment:
```bash
conda env create -f environment.yml
conda activate wormlab3d
```

## Configuration

1. Copy `.env.sample` to `.env`
2. Update database credentials in `.env`


## Running Midline finder

Execute the main analysis:
```bash
python scripts/midlines3d/mf_trial.py --argsfile=params/mf_trial.txt
```

### Parameters
- Parameters file: `params/mf_trial.txt`
- Modify trial settings, frame ranges, and optimization parameters in this file



