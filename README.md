# Policy Classifier Lightweight Replication

This repository contains a lightweight replication package for the report:

**Replicating an NLP-Based Policy Classification Study: A Critical Assessment of Incentive Classification for Policy Analysis**

The replication is based on the public materials released for:

Waskow, M.A. and McCrae, J.P. (2025) *Enhancing Policy Analysis with NLP: A Reproducible Approach to Incentive Classification*.

## Purpose

This package is designed to meet the coursework requirement that all replication code can be run with a single execution on another machine. It provides a simplified offline baseline rather than a full reproduction of the paper’s final transformer-embedding pipeline.

## Included files

The repository includes the following files required for one-command execution:

- `run_replication.sh`
- `replication_light.py`
- `19Jan25_firstdatarev.json`
- `27Jan25_query_checked.json`
- `requirements.txt`

It may also include generated output files:

- `merged_dataset.csv`
- `replication_results.json`
- `replication_summary.txt`

## Install dependencies

Install the required Python packages with:

```bash
pip install -r requirements.txt
