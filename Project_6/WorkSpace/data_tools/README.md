# More labelled training data for Challenge 1 (C-UAS detection and tracking)

These scripts build a bigger `train/` set in the exact challenge format, with no hand labelling.
`valid/` and `test/` are never touched, so scores stay comparable with earlier runs.

Why: the challenge train split has only 12 labelled clips (9 drone, 2 airplane, 2 helicopter)
and no airplane under 32 px, while 28 of the 36 validation airplanes are under 32 px.

## Quick start

Run from the repository root. `"Project 6/dataset"` is the challenge dataset as it sits in this
repo; change it if yours is elsewhere. Needs Python 3.10+, `pillow` and `numpy`.

```bash
# 1. 800 synthetic images: real train targets, shrunk to 6-32 px, pasted on other train frames
python Project_6/WorkSpace/data_tools/make_synthetic.py --dataset-dir "Project 6/dataset" --out-dir Project_6/WorkSpace/extra_data/synth --n 800

# 2. Public drone dataset: first look at what it contains (downloads 11 MB of labels)
python Project_6/WorkSpace/data_tools/get_public_drones.py --analyze

# 3. Download one part (0.9 GB) and convert 3,000 images whose largest drone is 64 px or less
python Project_6/WorkSpace/data_tools/get_public_drones.py --parts test --out-dir Project_6/WorkSpace/extra_data/seraphim --max-images 3000

# 4. Build the new dataset: challenge train + the two extra sets
python Project_6/WorkSpace/data_tools/challenge_data.py merge --base "Project 6/dataset" --add Project_6/WorkSpace/extra_data/synth --add Project_6/WorkSpace/extra_data/seraphim --out Project_6/WorkSpace/dataset_plus
```

Then train exactly as before, pointing at `Project_6/WorkSpace/dataset_plus` instead of the
challenge `dataset/`. If you build native-resolution crops first, run that step on
`dataset_plus/train`.

Step 1 should print about 800 images and 1,800 boxes. Step 4 prints a table of boxes per class
and per size, and ends with `format OK`.

Everything generated lands in `extra_data/`, `dataset_plus/` and `_downloads/`. These folders
are in `.gitignore`: rebuild them, do not push them.

## Status: what is tested and what is not

| Script | Tested on |
|---|---|
| `make_synthetic.py` | The real challenge train split. The 800 images were checked by eye. |
| `challenge_data.py` | The real challenge data, every command. |
| `get_public_drones.py` | Zip files built locally in the layout the dataset page describes. **Not yet run against the real download.** |

So run step 2 before step 3. If step 2 prints image counts, the label format is as expected.
If step 3 reports `no label file` for most images, the file layout differs from the dataset
page: send the output back before going further.

Nothing here has been trained on yet. Whether each extra set helps is still to be measured on
`valid/`.

## What each script does

**`make_synthetic.py`** cuts real targets out of train frames where they sit on a smooth
background, shrinks them, and pastes them on other train frames. The label is exact because the
script places the target. It only ever shrinks a target, and it skips a spot where the target
would not be visible. Limit: all airplanes come from one clip and helicopters from 32 cut-outs,
so it adds sizes and backgrounds, not new aircraft types. `--bg-dir` adds your own background
images, which must contain no aircraft.

**`get_public_drones.py`** downloads the Seraphim drone dataset from Hugging Face (83,483 images
at 640x640, one class, labels included, no account needed) and converts it. Every box becomes
`drone_air`. Parts: `test` is 0.9 GB with 8,349 images; `train1` to `train4` are 1.5 to 2.6 GB
each. Use `--parts test,train1` and a higher `--max-images` for more. `--max-side 0` also keeps
large drones.

**`challenge_data.py`** has five commands:

| Command | Use |
|---|---|
| `check <folder>` | Validate a folder and print boxes per class and size. |
| `convert` | Turn any COCO export (Roboflow, CVAT) into the challenge format. It prints the class mapping; check it. |
| `from-mot` | Turn a video plus a tracker output file into labelled frames. With `--min-track-len 15 --force-class drone_air` it auto-labels your own drone videos and drops short false alarms. |
| `negatives` | Wrap a folder of images with no aircraft (sky, birds, wires) as extra negatives. |
| `merge` | Build the new dataset from the challenge dataset plus any number of `--add` folders. |

Run any script with `--help` for all options.

## Rules

- Extra data goes into `train/` only. Do not train on the challenge validation or test images
  or videos. `merge --final-include-valid` exists for the very last model only, once every
  setting is frozen.
- Add one source at a time and score on `valid/` with the official scorer. Keep a source only
  if it helps.
- Look at about 50 images of any public source before training on it. Seraphim was collected
  from 23 community datasets, so label quality varies.
- Label convention of the challenge: tight boxes, fixed-wing drones are `drone_air`, birds get
  no box.

## Credit

Seraphim drone detection dataset, licence CC BY 4.0:
https://huggingface.co/datasets/lgrzybowski/seraphim-drone-detection-dataset
Credit it in the presentation if it is used.
