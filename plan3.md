# Plan 3: Deep Fusion (LLaVA + LLaVA) - Implementation Plan
**"The Pi0-Style Lockstep Architecture"**

We will implement a **Deep Fusion** architecture where a separate Action Expert (same architecture as the VLM) runs in parallel with the VLM backbone. They share information layer-by-layer via **Cross-Attention** or **Concatenated Attention** (depending on the exact Pi0 interpretation, usually it's Cross-Attention where Expert attends to VLM's KV cache at each layer).

## 1. Core Concept: `DeepFusionBlock`
Instead of standard Transformer blocks, we need a mechanism that allows the Expert model to "peek" into the VLM's internal state at every depth.

- **VLM (Perception):** Runs standard `LlamaModel` (LLaVA-3D). It produces `hidden_states` at every layer $L$.
- **Expert (Action):** Runs a modified `LlamaModel`. At Layer $L$, its Self-Attention module not only attends to its own tokens (Time + Noisy Actions) but also has access to the VLM's keys/values from Layer $L$.

## 2. Required Artifacts

### A. New Module: `model/modeling_deep_fusion.py`
We will create a specialized model definition for the Expert that supports deep fusion. We cannot reuse standard HF `LlamaModel` easily because we need to inject the VLM's intermediate states into the Expert's forward pass.

**Key Classes:**
1.  **`DeepFusionLlamaAttention`**:
    -   Inherits/Copies from `LlamaAttention`.
    -   **Modification:** The `forward` method will accept an extra argument: `cross_key_value_states` (The VLM's KV states for this layer).
    -   **Logic:** It will concatenate the Expert's KV (Time/Action) with the VLM's KV (Image/Text) *before* computing attention scores. This effectively allows the Action tokens to attend to the VLM context at that specific abstraction level.

2.  **`DeepFusionLlamaDecoderLayer`**:
    -   Wraps the `DeepFusionLlamaAttention` and the standard MLP.
    -   Passes the `cross_key_value_states` down to the attention layer.

3.  **`DeepFusionLlamaModel`**:
    -   The main body of the Expert.
    -   **Forward Pass Change:** It must accept a list/tuple of `vlm_hidden_states` (one per layer) from the main VLM.
    -   It iterates through its own layers and the VLM's layers in sync, passing the corresponding VLM state into the Expert layer.

4.  **`DeepFusionActionExpert`**:
    -   The high-level wrapper (similar to `FlowMatchingActionExpert` but for Plan 3).
    -   **Backbone:** `DeepFusionLlamaModel` (instantiated with same config as VLM).
    -   **Logic:**
        -   `forward(vlm_outputs, actions, time)`:
            1.  Extracts `hidden_states` from `vlm_outputs`.
            2.  Runs the Deep Fusion Llama backbone.
            3.  Projects output to velocity.

### B. Updated Main Model: `model/modeling_spatialvla.py`
We need to update `SpatialVLAForConditionalGeneration` to support this new mode.

-   **Initialization:**
    -   Add support for `deep_fusion_config` (or just detecting if we are in Plan 3 mode).
    -   Initialize `self.action_expert` as `DeepFusionActionExpert` instead of the Late Fusion one if configured.
-   **Forward Pass:**
    -   **Crucial Change:** We must run the VLM with `output_hidden_states=True` to get the intermediate states for *every layer*.
    -   Pass `outputs.hidden_states` (all layers) to `self.action_expert` instead of just the last one.

## 3. Implementation Steps

1.  **Step 1: Create `model/modeling_deep_fusion.py`**:
    -   Copy relevant Llama code (Attention, Layer, Model) from `transformers` (to ensure compatibility and access to internals).
    -   Modify `Attention` to handle `cross_key_value_states` (Concatenation strategy).
    -   Implement the `DeepFusionActionExpert` wrapper.

2.  **Step 2: Update `model/configuration_spatialvla.py`**:
    -   Add `deep_fusion: bool` flag or `action_expert_type` enum ("late_gemma", "deep_llama").

3.  **Step 3: Update `model/modeling_spatialvla.py`**:
    -   Update `forward` and `predict_action` to extract all hidden states and pass them to the expert if in Deep Fusion mode.

## 4. Key Challenges & Solutions

-   **Dimension Mismatch:** The VLM and Expert must have the same `hidden_size` for simple Deep Fusion (or we need linear projections at *every layer*).
    -   *Solution:* We will enforce that the Expert Config matches the VLM Text Config (e.g., both 7B Llama, or both 1B Llama).
-   **Memory Usage:** Storing all KV caches for two models is expensive.
    -   *Mitigation:* Gradient Checkpointing will be essential for training.
-   **VLM Layer Sync:** We assume 1-to-1 mapping of layers. If VLM has 32 layers, Expert should have 32 layers.

## 5. Summary of Data Flow
1.  **VLM:** `[Img, Text]` -> Layer 0 -> ... -> Layer N -> `[H_0, ..., H_N]`
2.  **Expert:** `[Time, Action]`
    -   Layer 0: Attends to (`[Time, Action]` KVs + `H_0` KVs)
    -   ...
    -   Layer N: Attends to (`[Time, Action]` KVs + `H_N` KVs)
3.  **Output:** Refined Action Velocity.
