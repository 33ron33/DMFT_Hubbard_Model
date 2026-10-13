#!/bin/bash
#SBATCH --job-name=cthyb-U8-n07-b25
#SBATCH --partition=campus
#SBATCH --account=acf-utk0014
#SBATCH --qos=campus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Project directory on ISAAC. Put doped_bethe_cthyb_dmft_new.py here.
WORKDIR="/lustre/isaac24/scratch/rnain/DMFT_with_Doping"
SCRIPT="$WORKDIR/doped_bethe_cthyb_dmft_new.py"
RUN_TAG="U8_n0p7_beta25_refined"
OUTPUT_ROOT="$WORKDIR/cthyb_test_results"
LOG_DIR="$OUTPUT_ROOT/$RUN_TAG"

# Load the modules required by YOUR ISAAC TRIQS installation.
# Do not add the DCA++-specific HDF5 LD_LIBRARY_PATH unless TRIQS also needs it.
module load gcc/13.4.0 openmpi/4.1.8-gcc13 fftw/3.3.11-gcc13 openblas/0.3.12

# IMPORTANT: Activate the Python environment containing triqs and triqs_cthyb.
# If using Conda on ISAAC, uncomment and adjust the following two lines:
# source "$HOME/miniforge3/etc/profile.d/conda.sh"
# conda activate /lustre/isaac24/scratch/rnain/envs/NN_Solver/
# Alternatively, select your TRIQS Python executable with PYTHON_BIN.
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: Cannot find Python script at: $SCRIPT" >&2
    exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python executable '$PYTHON_BIN' is unavailable. Activate your ISAAC TRIQS environment." >&2
    exit 1
fi

cd "$WORKDIR"
mkdir -p "$LOG_DIR"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$LOG_DIR/.matplotlib"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR"

# Fail early rather than wasting an allocation with an incorrect Python.
"$PYTHON_BIN" - <<'PY'
import sys
print('Python:', sys.executable, flush=True)
try:
    from triqs_cthyb import Solver
    from triqs.gf import MeshImFreq
except ImportError as exc:
    raise SystemExit(
        'ERROR: This ISAAC Python does not have a usable TRIQS/CTHYB installation: '
        + str(exc)
        + '\nActivate the Linux TRIQS environment and resubmit.'
    )
print('TRIQS/CTHYB imports: OK', flush=True)
PY

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Run tag: $RUN_TAG"
echo "Start: $(date)"

"$PYTHON_BIN" -u "$SCRIPT" \
  --u-over-t 8 \
  --fillings 0.7 \
  --beta-t 25 \
  --t 0.25 \
  --n-iw 256 \
  --n-tau 2049 \
  --n-l 50 \
  --n-cycles 100000 \
  --n-warmup-cycles 10000 \
  --length-cycle 600 \
  --final-replicas 4 \
  --final-n-cycles 200000 \
  --final-n-warmup-cycles 10000 \
  --final-length-cycle 800 \
  --seed 271828 \
  --vary-seed-by-iteration \
  --cthyb-verbosity 1 \
  --cthyb-max-time -1 \
  --fit-min-w-over-t 12 \
  --fit-max-w-over-t 20 \
  --fit-max-moment 3 \
  --mix 0.30 \
  --dmft-tol 2e-3 \
  --density-tol 1e-3 \
  --density-stability-tol 1e-3 \
  --mu-tol-over-t 1e-3 \
  --stable-iterations 2 \
  --min-dmft-iter 4 \
  --max-dmft-iter 60 \
  --max-mu-iter 16 \
  --mu-min-over-t 0.50 \
  --mu-max-over-t 0.65 \
  --conv-n-iw 32 \
  --max-bracket-expansions 4 \
  --plot-n-iw 80 \
  --output-dir "$OUTPUT_ROOT" \
  --tag "$RUN_TAG" \
  2>&1 | tee "$LOG_DIR/cthyb_${SLURM_JOB_ID:-manual}.log"

echo "Finished: $(date)"
