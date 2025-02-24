# WaterDiffusion: Learning a Prior-involved Unrolling Diffusion for Joint Underwater Saliency Detection and Visual Restoration

## Introduction
Underwater salient object detection (USOD) plays a pivotal role in various vision-based marine exploration tasks. However, existing USOD techniques face the dilemma of object mislocalization and imprecise boundaries due to the complex underwater environment. The quality degradation of raw underwater images (caused by selective absorption and medium scattering) makes it challenging to perform instance detection directly. One conceivable approach involves initially removing visual disturbances through underwater image enhancement (UIE), followed by saliency detection. However, this two-stage approach neglects the potential positive impact of the restoration procedure on saliency detection due to it executes in a cascade. Based on this insight, we propose a generalized prior-involved diffusion model, called WaterDiffusion for collaborative underwater saliency detection and visual restoration. Specifically, we first propose a revised self-attention joint diffusion, which embeds dynamic saliency masks into the diffusive network as latent features. By extending the underwater degradation prior into the multi-scale decoder, we innovatively exploit optical transmission maps to aid in localizing underwater salient objects. Then, we further design a gate-guided binary indicator to select either normalized or original channels for improving feature generalization. Finally, the Half-quadratic Splitting is introduced into the unfolding sampling to refine saliency masks iteratively. Comprehensive experiments demonstrate the superior performance of WaterDiffusion over state-of-the-art methods in both quantitative and qualitative evaluations.

![image](https://github.com/user-attachments/assets/e1250a0c-462b-48be-9df2-aa1e3213adf4)

## Requirement
* Python 3.8
* Pytorch 2.0.1
* CUDA 11.8

## Training
 Python train.py

## Testing
 Python test.py

## Notes
We will upload the Full code here in the future. Thanks for your attention!

If you encounter any difficulties in the testing process, please feel free to contact us at changlb666@163.com.
