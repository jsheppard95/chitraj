#!/bin/bash
set -euo pipefail

MODE="${1:-gtn_sample}"
CHUNK_SIZE="${2:-100}"
MAX_CHUNK_ID="${3:-299}"
CHUNK_DIR="${4:-chunk_output}"

case "$MODE" in
    r1_nosample)
        MODE_TAG="R1_no_sample"
        ;;
    r1_sample)
        MODE_TAG="R1_sample_5000"
        ;;
    gtn_sample)
        MODE_TAG="GTN_sample_5000"
        ;;
    *)
        echo "Unknown MODE: $MODE"
        echo "Use one of: r1_nosample, r1_sample, gtn_sample"
        exit 1
        ;;
esac

MISSING_FILE="missing_chunks_${MODE_TAG}.txt"
: > "$MISSING_FILE"

for chunk_id in $(seq 0 "$MAX_CHUNK_ID"); do
    summary_file="${CHUNK_DIR}/${MODE_TAG}_chunk${chunk_id}_per_frame_summary.csv"
    raw_file="${CHUNK_DIR}/${MODE_TAG}_chunk${chunk_id}_raw_distributions.npz"

    if [[ ! -f "$summary_file" || ! -f "$raw_file" ]]; then
        echo "$chunk_id" >> "$MISSING_FILE"
    fi
done

N_MISSING=$(wc -l < "$MISSING_FILE")

if [[ "$N_MISSING" -eq 0 ]]; then
    echo "No missing chunks found for ${MODE_TAG}."
    rm -f "$MISSING_FILE"
    exit 0
fi

echo "Found $N_MISSING missing chunks for ${MODE_TAG}:"
cat "$MISSING_FILE"

ARRAY_MAX=$((N_MISSING - 1))

sbatch \
    --array=0-"${ARRAY_MAX}" \
    --export=ALL,MODE="${MODE}",CHUNK_SIZE="${CHUNK_SIZE}",MISSING_FILE="${MISSING_FILE}",CHUNK_DIR="${CHUNK_DIR}" \
    rerun_missing_chunks.sbatch