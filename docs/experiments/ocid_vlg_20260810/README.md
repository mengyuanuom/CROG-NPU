# OCID-VLG evaluation - 2026-08-10

This report records the OCID-VLG test-set evaluation run on server commit
`91d3c51ab5c6a8964c5b4357675d7d3b3def35b6`.

## Scope

- Dataset split: `test`, 17,749 referring-expression samples.
- Protocol: `crog_legacy`.
- Hardware: 8 Ascend 910B3 NPUs, split into two concurrent four-NPU jobs.
- Loader: batch size 32 and 2 workers per NPU.
- Visualization: disabled.
- The evaluator shards samples without padding and sums all statistics across
  ranks before reporting metrics.

Only DROG and DROG-OFF had trained OCID-VLG checkpoints on the server. CROG,
LGD, GGCNNCLIP, GRConvNetCLIP, and ETRG could not be evaluated because no
checkpoint existed under `exp/OCID-VLG`.

Before evaluation, the eight NPUs were occupied by `npu_resource_filler.py`.
That dedicated filler process and its orphaned workers were stopped. No NPU
process remained after both evaluations completed.

## Selected checkpoints

| Model | Epoch | Logged validation IoU | Logged validation J@1 | Logged validation J@5 | Logged width decode |
| --- | ---: | ---: | ---: | ---: | --- |
| DROG-OFF | 30 | 82.11 | 91.24 | 94.12 | sigmoid (matched) |
| DROG | 36 | 82.31 | 85.78 | 93.57 | sigmoid (mismatched) |

The training logs report `world_size: 1` and batch size 32 for both models.
Both 36-epoch runs completed without an exception. Training time was 1 day
18:57:23 for DROG-OFF and 1 day 18:05:52 for DROG.

## Test results

| Model | IoU | Pr@50 | Pr@60 | Pr@70 | Pr@80 | Pr@90 | J@1 | J@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DROG-OFF | 81.23 | 97.02 | 95.66 | 89.19 | 68.54 | 22.33 | **89.30** | 92.94 |
| DROG | **81.38** | **97.39** | **96.40** | **89.67** | 67.82 | 22.03 | 89.10 | **93.58** |

DROG-OFF leads test J@1 by only 0.20 points, while DROG leads IoU by 0.15 and
J@5 by 0.64 points. The original DROG validation number is not directly
comparable because it used a mismatched sigmoid width decoder.

## DROG width-decoder A/B

The DROG checkpoint was saved with `grasp_size_activation=None`. The
`DROG.grasp_size_loss_activation = "clamp"` declaration was added after this
training run finished. At training time, the missing YAML field therefore fell
back to the evaluator's legacy sigmoid default even though DROG trained its raw
width output directly against the normalized target.

The same checkpoint was re-evaluated twice on the same 8,669-sample validation
split with current code. Only the width activation changed:

| Width decode | IoU | J@1 | J@5 |
| --- | ---: | ---: | ---: |
| clamp (matches DROG loss) | 82.31 | **89.62** | **94.50** |
| sigmoid (legacy mismatch) | 82.31 | 85.78 | 93.57 |

The sigmoid run exactly reproduces the historical training-log result, while
IoU remains identical. This isolates the apparent test-set gain to width
decoding rather than segmentation or an easier test split. The correct aligned
comparison is validation clamp 89.62/94.50 versus test clamp 89.10/93.58:
the test split is lower by 0.52 J@1 and 0.92 J@5. DROG-OFF remains sigmoid
because its width loss applies sigmoid during training.

## Checkpoint identity

```text
1b18ea348918854eac60ad55294f518e06a6b05613feeb946a202a37de0950e3  best_epoch_030_J1_91.24_J5_94.12.pth
93462598fe0967e256b034a843bc4fa45c64d8f223385777fa6591b01e47dd06  best_epoch_036_J1_85.78_J5_93.57.pth
```

## Reproduction

Load the matching CANN environment first, then run one model per four-NPU
group. The model YAML files now define test batch size 32, 2 workers, and
metadata-driven grasp-size decoding.

```bash
source /data1/ma00959358/cann_path/cann-8.5.0/set_env.sh

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29611 \
  test_crog.py --config config/OCID-VLG/drogoff.yaml --opts \
  DATA.root_path /data1/ma00959358/pangu/CROG-NPU/datasets/OCID-VLG \
  TRAIN.resume exp/OCID-VLG/drogoff_ocid_vlg_8npu_20260801_001051_484/best_epoch_030_J1_91.24_J5_94.12.pth

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 torchrun \
  --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29613 \
  test_crog.py --config config/OCID-VLG/drog.yaml --opts \
  DATA.root_path /data1/ma00959358/pangu/CROG-NPU/datasets/OCID-VLG \
  TRAIN.resume exp/OCID-VLG/drog_ocid_vlg_8npu_20260801_001106_814/best_epoch_036_J1_85.78_J5_93.57.pth
```

The first DROG launch exposed missing TEST keys in `drog.yaml`; it stopped
before model or NPU initialization. The successful run used equivalent
effective settings from the complete DROG-OFF profile with the architecture
overridden to DROG. The configuration fix included with this report makes the
clean DROG command above directly reproducible.

Raw rank-0 logs are stored beside this report as `drogoff_test.txt`,
`drog_test.txt`, `drog_val_clamp.txt`, and `drog_val_sigmoid.txt`.
