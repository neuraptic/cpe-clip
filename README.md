# CPE-CLIP
Official implementation of the ["*Multimodal Parameter-Efficient Few-Shot Class Incremental Learning*"](https://arxiv.org/abs/2303.04751) paper.

### About
CPE-CLIP is a CLIP-based Continual Learning model architecture showing promising performance in Few-Shot Class Incremental Learning by leveraging prompt-tuning and a time-varying regularizer.

### Installation

Clone the repo and install the requirements with:

```
pip install -r requirements.txt
```

Other requirements:
- Python 3.10.6
- PyTorch >= 1.12.1
- CUDA >= 11.4

### Model
The code for the CLIP model adapted for parameter-efficient learning used in the paper is located in `/CoLeLib/models/clip_models` and contains:
- **CLIPTextModelForPromptTuning**: a modified CLIP Language Encoder that accepts learnable prompts to prepend to tokens layerwise.
- **CLIPVisionModelFroPromptTuning**: a modified CLIP Vision Encoder that accepts learnable prompts to append to patch embeddings layerwise.
- **CLIPParameterEfficient**: the final wrapper combining both the encoders and applying parameter-efficient contrastive-learning.  

### Training
The code for the Continual Learning adaptation and training setup is located in `CoLeLib/training/strategies/clip_pe.py`.
