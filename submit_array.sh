#!/bin/bash
#SBATCH --job-name=chilife_chunk
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-299
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-user=sheppard@ucsb.edu
#SBATCH --mail-type=all                # Send email at job start and job end

set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate chilife_env

mkdir -p logs chunk_output

CHUNK_SIZE=100
START_FRAME=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))

# choose one:
# MODE="${MODE:-r1_nosample}"
# MODE="${MODE:-r1_sample}"
MODE="${MODE:-gtn_sample}"

TOP1="./tip3p/JS_1bar_equib.pdb"
TRAJ1="./tip3p/JS_1bar.xtc"
TOP2="./tip3p/JS_3kbar_equib.pdb"
TRAJ2="./tip3p/JS_3kbar.xtc"

SITE1_COND1=406
SITE2_COND1=537
SITE1_COND2=4
SITE2_COND2=135

LABEL_NAME=""
SAMPLE_ARG=()
ROTLIB_ARG=()
MODE_TAG=""

case "$MODE" in
    r1_nosample)
        LABEL_NAME="R1M"
        MODE_TAG="R1_no_sample"
        ;;
    r1_sample)
        LABEL_NAME="R1M"
        SAMPLE_ARG=(--sample 5000)
        MODE_TAG="R1_sample_5000"
        ;;
    gtn_sample)
        LABEL_NAME="GTN"
        SAMPLE_ARG=(--sample 5000)
        ROTLIB_ARG=(--rotlib GTN_rotlib.npz)
        MODE_TAG="GTN_sample_5000"
        ;;
    *)
        echo "Unknown MODE: $MODE"
        exit 1
        ;;
esac

python compare_R1_rotlib_traj.py \
    --top1 "$TOP1" \
    --traj1 "$TRAJ1" \
    --top2 "$TOP2" \
    --traj2 "$TRAJ2" \
    --site1-cond1 "$SITE1_COND1" \
    --site2-cond1 "$SITE2_COND1" \
    --site1-cond2 "$SITE1_COND2" \
    --site2-cond2 "$SITE2_COND2" \
    --label-name "$LABEL_NAME" \
    "${SAMPLE_ARG[@]}" \
    "${ROTLIB_ARG[@]}" \
    --start-frame "$START_FRAME" \
    --max-frames "$CHUNK_SIZE" \
    --stride 1 \
    --outdir chunk_output \
    --tag "${MODE_TAG}_chunk${SLURM_ARRAY_TASK_ID}"