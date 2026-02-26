# Pretraining Codebase for AnyPPG and Baselines

This directory contains the source code for the pretraining experiments presented in the paper, including the proposed **AnyPPG** method and the self-supervised learning baselines: **SimCLR** and **BYOL**.

## Overview

- **AnyPPG**: The proposed method utilizing PPG-ECG CLIP-based pretraining.
- **SimCLR**: A contrastive learning baseline implementation.
- **BYOL**: A self-supervised learning baseline implementation (Bootstrap Your Own Latent).

## Directory Structure

- `anyppg/`: Contains the implementation of the AnyPPG method.
- `simclr/`: Contains the implementation of the SimCLR baseline.
- `byol/`: Contains the implementation of the BYOL baseline.

## Data Preprocessing

For details regarding the specific processing of the pretraining datasets, please refer to the `AnyPPG/preprocessing` directory.
