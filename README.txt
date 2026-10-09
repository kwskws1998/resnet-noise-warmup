ResNet-18 noise warm-up: full-data scale-up screen

STATUS
This is an execution package, not a completed CUDA experiment.
The earlier Mac PoC found calibration/acquisition signals, but no consistent
forgetting improvement. Full-data performance and novelty are unverified.
verification.json records CPU synthetic integration checks, not GPU results.

ENVIRONMENT
Use one NVIDIA GPU (RTX 3090-class, 24 GB), Linux, Python 3.10 or newer,
a CUDA-enabled PyTorch 2.8.0 environment, at least 16 GB host RAM, and
at least 10 GB free persistent disk. A 20 GB volume provides more margin.
Torchvision 0.23.0 must match torch 2.8.0. Prefer an existing compatible
CUDA PyTorch image. Installation in a fresh environment:

python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

CUDA availability must print True. Do not launch full training on CPU.
The runner fails instead of silently falling back to CPU.
No shell scripts, Docker credentials, API tokens, data, or checkpoints are
included in the source package. All paths are relative to these Python files.

PROTOCOL
2 seeds (1, 2), 3 conditions, exactly 6 runs:
  none: no noise warm-up
  full: all-parameter random-input/random-label warm-up
  full_var: same warm-up plus pooled-feature variance floor

CIFAR-10: official 50000 training images split into 45000 training and 5000
validation. Official 10000 test images are stored separately. The previous
PoC already examined 1000 of these test images; final results report both
all 10000 and the 9000 excluded from that earlier test subset. This is not
a claim that all official test data have never been seen before.

ResNet-18 from scratch, CIFAR stem, no initial maxpool, shared 10-way head.
Three sequential domains: 0, 90, 180 degree rotations of the same examples.
Domain-incremental, no replay, no task identity at inference.
50 epochs per domain, real batch size 128, Adam LR 0.001, zero weight decay,
constant LR, no augmentation, optimizer reset at every domain boundary.
FP32 with TF32 disabled; no AMP in this replication screen.
Warm-up: 200 updates at batch size 64, Adam LR 0.001, fresh Gaussian inputs
and independent uniform random labels. Feature-SD floor is 80% of the
initial Gaussian-probe feature SD; squared positive deficit penalty weight 1.
The variance penalty is used ONLY during warm-up, not real-data training.
All conditions reset and recalibrate BN buffers on the same 640 unlabeled
task-0 training inputs. Learned BN affine parameters are retained.

This expands data and training duration while retaining the PoC optimizer
and intervention. It is not a tuned CIFAR-10 accuracy benchmark. The real
batch size changes from 64 to 128; warm-up sample exposure remains unchanged.
Data size, duration, real batch size, and hardware differ from the Mac PoC.
Do not interpret changes as an isolated architecture or dataset-size effect.

RUN ORDER (execute from this directory)
1. Hardware/operator check and measured compute estimate:

python run.py smoke
python run.py benchmark

results/benchmark.json reports training-step timing and training-only hours
for all six runs. This excludes validation, checkpoint IO, preparation,
data movement/indexing, provisioning and idle time. It is not a bill quote.
The synthetic benchmark VRAM excludes resident full data.

2. Prepare the verified public dataset:

python prepare.py

This downloads the archive and checks the fixed SHA-256. If you already
have this exact archive, use python prepare.py --archive /path/to/cifar10_fastai.tgz
Preparation separates train/validation/test files and stores file memberships.
Training opens train and validation only. Test pixels are decoded/stored
during preparation but test prediction metrics remain deferred.

3. Execute the first seed, then the second, without changing the protocol:

python run.py train --seeds 1
python run.py train --seeds 2

Alternatively: python run.py train
Use --conditions none full full_var to select a subset for operational reasons.
The full planned grid must complete before final test evaluation.
Progress and validation metrics print at each epoch; each completed epoch
atomically saves latest.pt including optimizer and loop state. Rerunning
the same command resumes. Every task endpoint is saved for final forgetting
evaluation. A persistent volume is required to survive machine deletion.
Do not run two workers against the same output directory simultaneously.

4. Finalize only after all six runs are complete and the protocol is frozen:

python run.py finalize

This evaluates task-end checkpoints on official test data. It refuses an
incomplete grid. A test_accessed.json marker prevents subsequent training
in that output directory. If evaluation is interrupted, rerun finalize.
Do not tune or increase seeds based on the finalized test results.
Stop/terminate the rented instance yourself when finished; this code does
not create cloud instances or control cloud billing.

RESULTS
results/validation_summary.csv: one row per completed condition/seed.
results/test_summary.csv: endpoints on all 10000 official test examples.
results/fresh_test_summary.csv: endpoints on the 9000 not used in the PoC test.
results/seed_XX_CONDITION/validation.json: full validation acquisition curves.
results/seed_XX_CONDITION/test_logits.npz: raw final evaluation predictions.
results/provenance.json: source/config/data hashes and runtime versions.

Metrics: final mean accuracy, old-domain final accuracy, forgetting,
first-domain accuracy/ECE/NLL, and validation acquisition AUC.
Forgetting averages (best task-end accuracy - final accuracy) over old tasks.
Do not infer retention improvement solely from a smaller forgetting number
when acquisition accuracy is lower. AUC includes epoch zero and is normalized
by training duration. Raw table values are fractions, not percentages.
Two seeds support screening; no p-values or strong significance claims.

DECISION AFTER THIS SCREEN
If retention improves in both seeds without worse acquisition, next compare
learning-rate/label-smoothing and matched-extra-update controls, then a
separately frozen class-incremental benchmark with a replay baseline.
If only ECE improves, investigate calibration/initialization, not a forgetting
claim. If the gap vanishes by epoch 50, interpret it as a transient or short-
training effect rather than evidence of durable retention improvement.
Keep the next experiment's test held out and audit novelty separately.

LOCAL VERIFICATION
python verify.py

This uses a real ResNet with tiny synthetic data to exercise all six run
paths, exact epoch resume, paired initialization, complete-grid test gate,
and recomputation of fresh-subset metrics. It does not require a GPU and
does not execute full CIFAR training. CUDA validation is still required.

GITHUB
Create an empty repository (private is appropriate for unpublished work),
then add these code/config files. .gitignore excludes data and checkpoints.
The folder contains no account credentials or absolute user-machine paths.
