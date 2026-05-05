# DFM Training for Low-Latency VisoMaster Streaming

This guide is for consent-based one-to-one DFM training where live latency and
occlusion quality matter.

## Recommended First Target

Use a 256 resolution DFM first. On an RTX 5090, the GPU is strong enough to try
larger models, but the live pipeline also pays for face detection, masking,
enhancement, compositing, and OBS/virtual-camera output. Increase resolution
only after the full live path is stable.

```text
First pass: 720p30, DFM 256, enhancer off, occlusion/XSeg on
Second pass: 720p30, DFM 256, lightweight enhancer on
Third pass: 1080p30, DFM 256 or 320
Last pass: 60 FPS or larger DFM
```

## Data Capture

Capture both identities with the same consent rules and production lighting you
expect to use on stream. Include normal speech, smiles, blinking, head turns,
and the hard cases that usually break masks.

Record extra clips for:

- hand over mouth
- hand over nose
- fingers crossing cheeks
- chin resting on hand
- hair crossing face
- glasses and reflections
- microphone or cup near mouth

## Why XSeg Matters

The DFM model improves the identity match. XSeg and occlusion masks decide what
pixels are allowed to be replaced. When your hand crosses your face, good output
depends on keeping original hand pixels in front of the generated face.

The final compositor behavior should be:

```text
swap only visible face skin/lips/nose regions
preserve foreground hand/object pixels
smooth mask boundaries over time
```

## VisoMaster Import Path

VisoMaster scans this folder:

```text
model_assets/dfm_models
```

Use:

```powershell
python tools/dfm_training/install_dfm_model.py --model D:\path\to\model.dfm
```

Then restart VisoMaster and choose `DeepFaceLive (DFM)`.

## Practical Tuning Order

1. Test the DFM without enhancer.
2. Turn on XSeg/occlusion masking.
3. Tune RCT/color options.
4. Add a lightweight face enhancer.
5. Increase input/output resolution.
6. Increase DFM resolution only if latency still feels good.
