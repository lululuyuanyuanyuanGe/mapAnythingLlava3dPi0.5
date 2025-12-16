# Project Goal: SpatialVLA with 3D Capabilities and Flow Matching Action Expert

## Overview
Our research goal is to build a Vision-Language-Action (VLA) model with intrinsic 3D understanding capabilities.
The architecture consists of three main components:

1.  **3D Reconstruction**: Powered by **MapAnything**. This provides the model with geometric understanding of the environment. (Status: Integrated)
2.  **VLM Backbone**: **LLaVA-3D**. This is a Vision-Language Model trained on 3D data, serving as the core reasoning engine. (Status: Integrated)
3.  **Action Expert**: **Pi0 (v0.5)**. This component uses a **Flow Matching** architecture to generate continuous robot actions. (Status: **Pending Integration**)

## Current Task
We need to integrate the Pi0 Action Expert into the existing SpatialVLA pipeline.
- A PyTorch implementation of Pi0 exists in `openpi/models_pytorch/pi0_pytorch.py`.
- The current SpatialVLA (`model/modeling_spatialvla.py`) uses LLaVA-3D as its backbone.
- We need to bridge the Pi0 flow matching logic (currently designed for `PaliGemmaWithExpertModel`) with the SpatialVLA/LLaVA-3D architecture.

## Technical Objectives
- Analyze the `Pi0Pytorch` implementation and its dependencies (`PaliGemmaWithExpertModel`).
- Determine how to adapt the flow matching architecture (action embeddings, time embeddings, AdaRMS conditioning if used) to the LLaVA-3D backbone.
- Implement the integration in `model/modeling_spatialvla.py`, enabling both flow matching training (loss computation) and inference (`sample_actions`).
