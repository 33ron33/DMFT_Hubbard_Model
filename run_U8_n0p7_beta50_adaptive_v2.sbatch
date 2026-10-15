#!/bin/bash
#SBATCH --job-name=cthyb-U8-n09-b50
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
WORKDIR=/lustre/isaac24/scratch/rnain/DMFT_with_Doping
PYTHON_BIN=/lustre/isaac24/scratch/rnain/envs/NN_Solver/bin/python
SCRIPT="$WORKDIR/doped_bethe_cthyb_dmft_adaptive.py"
RUN_TAG=U8_n0p9_beta50_adaptive_retry
OUTPUT_ROOT="$WORKDIR/cthyb_test_results"
LOG_DIR="$OUTPUT_ROOT/$RUN_TAG"
module load gcc/13.4.0 openmpi/4.1.8-gcc13 fftw/3.3.11-gcc13 openblas/0.3.12
cd "$WORKDIR"
mkdir -p "$LOG_DIR/.matplotlib"
export MPLBACKEND=Agg MPLCONFIGDIR="$LOG_DIR/.matplotlib" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ -f "$SCRIPT" && -x "$PYTHON_BIN" ]] || { echo 'Missing Python or script' >&2; exit 1; }
"$PYTHON_BIN" -c 'from triqs_cthyb import Solver; from triqs.gf import MeshImFreq; print("TRIQS imports OK")'
echo "Starting: $(date); job ${SLURM_JOB_ID:-manual}; node $(hostname)"
"$PYTHON_BIN" -u "$SCRIPT" \
 --u-over-t 8 --fillings 0.9 --beta-t 50 --t 0.25 \
 --n-iw 256 --n-tau 2049 --n-l 50 \
 --coarse-n-cycles 20000 --coarse-n-warmup-cycles 4000 \
 --coarse-length-cycle 450 --coarse-dmft-tol 1e-2 \
 --coarse-density-stability-tol 1.5e-2 --coarse-max-dmft-iter 40 \
 --coarse-max-mu-iter 16 --switch-density-distance 0.06 \
 --n-cycles 50000 --n-warmup-cycles 10000 --length-cycle 800 \
 --final-replicas 6 --final-n-cycles 200000 \
 --final-n-warmup-cycles 40000 --final-length-cycle 1000 \
 --seed 314159 --vary-seed-by-iteration \
 --cthyb-verbosity 1 --cthyb-max-time -1 \
 --fit-min-w-over-t 12 --fit-max-w-over-t 20 --fit-max-moment 3 \
 --mix 0.25 --dmft-tol 3e-3 --density-tol 1e-3 \
 --density-stability-tol 8e-3 --mu-tol-over-t 5e-4 \
 --stable-iterations 2 --min-dmft-iter 6 --max-dmft-iter 60 \
 --max-mu-iter 20 --mu-min-over-t 1.30 --mu-max-over-t 1.65 \
 --mu-guess-over-t 1.43 --conv-n-iw 32 \
 --max-bracket-expansions 4 --plot-n-iw 80 \
 --output-dir "$OUTPUT_ROOT" --tag "$RUN_TAG" \
 2>&1 | tee "$LOG_DIR/cthyb_${SLURM_JOB_ID:-manual}.log"
echo "Finished: $(date)"
