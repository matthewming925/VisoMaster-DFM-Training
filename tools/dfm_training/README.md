# DFM Training Helpers for VisoMaster

These helpers prepare consent-based DeepFaceLab workspaces for training a
pair-specific DeepFaceLive/DFM model, then install the exported `.dfm` file into
VisoMaster.

The fast live-stream path is:

```text
Prepare source/destination frames
-> extract aligned faces in DeepFaceLab
-> train SAEHD or AMP
-> train/apply XSeg for hands, mouth, nose, and face-edge occlusion cases
-> export DFM
-> copy DFM into model_assets/dfm_models
-> select DeepFaceLive (DFM) in VisoMaster
```

Use this only with explicit permission from every person whose face, voice, or
likeness appears in the source or destination material.

## Files

- `prepare_dfm_workspace.py` creates a DeepFaceLab-style workspace from videos
  and/or image folders.
- `install_dfm_model.py` validates and copies an exported `.dfm` or `.onnx`
  model into VisoMaster's `model_assets/dfm_models` directory.
- `dfm_training_config.example.json` is a repeatable config template.

## Prepare Data

Install `ffmpeg` first if you want to extract frames from videos.

Example with one source video and one destination/live-reference video:

```powershell
python tools/dfm_training/prepare_dfm_workspace.py `
  --workspace D:\dfm_workspaces\my_pair `
  --src-video D:\capture\source_identity.mp4 `
  --dst-video D:\capture\streaming_actor.mp4 `
  --src-fps 8 `
  --dst-fps 8 `
  --max-width 1920 `
  --profile-name my_pair
```

Example with image folders:

```powershell
python tools/dfm_training/prepare_dfm_workspace.py `
  --workspace D:\dfm_workspaces\my_pair `
  --src-images D:\capture\source_photos `
  --dst-images D:\capture\target_photos `
  --profile-name my_pair
```

The script creates:

```text
workspace/
  data_src/
  data_dst/
  data_src/aligned/
  data_dst/aligned/
  model/
  mask_review/
  manifest.json
  deepfacelab_next_steps.md
```

## DeepFaceLab Training Checklist

Run the generated workspace through DeepFaceLab:

1. Extract source faces from `data_src`.
2. Extract destination faces from `data_dst`.
3. Sort and clean both aligned facesets.
4. Open XSeg editor and label hard mask frames.
5. Train XSeg and apply trained XSeg masks to source and destination faces.
6. Train SAEHD or AMP.
7. Export SAEHD or AMP as `.dfm`.

For live streaming, start with a 256 resolution DFM. Move to 320 or 384 only
after VisoMaster latency is stable on your RTX 5090.

## Masking Strategy for Hands Over Mouth or Nose

Mouth and nose region masks help, but the most important mask is a foreground
occlusion mask. When a hand covers the face, the compositor should keep the hand
pixels from the original frame and apply the swapped face only to the visible
skin region around it.

Prioritize XSeg review frames that include:

- fingers crossing lips or teeth
- hand covering the nose
- hand touching cheeks or chin
- hair across the face
- microphone, cup, phone, glasses, or other objects near the mouth
- fast motion blur around hands

The goal is not to mask the whole mouth permanently. The goal is to teach the
mask what should remain in front of the generated face.

## Install the Exported Model

After DeepFaceLab exports a `.dfm` model:

```powershell
python tools/dfm_training/install_dfm_model.py `
  --model D:\dfm_workspaces\my_pair\model\my_pair.dfm
```

The script copies the model into:

```text
model_assets/dfm_models/
```

Restart VisoMaster, choose `DeepFaceLive (DFM)` as the swap model, then select
your model from the DFM dropdown.

## Low-Latency VisoMaster Starting Point

For the first live test:

```text
Input/output: 720p30
Swap model: DeepFaceLive (DFM)
DFM resolution: 256
Face enhancer: off
DFL XSeg / occlusion mask: on
Color transfer / RCT: test on and off
```

Then add quality features one at a time:

```text
1. Enable GPEN 256 or lightweight restoration only if FPS stays stable.
2. Try 1080p30 after masks are stable at 720p30.
3. Try 60 FPS last.
```

## Notes

This folder does not include DeepFaceLab binaries, trained weights, datasets, or
any material for impersonation without consent. It only automates preparation,
documentation, and local model installation for VisoMaster.
