#!/bin/bash
# Fetch the MaleCNS v1.0 release files used by the graph prior.
#
# Google Cloud Storage throttles a single connection hard from some networks (we measured
# ~100 KB/s single-stream vs ~430 KB/s with 16 ranged connections), so the download is split
# into fixed-size ranged chunks fetched in parallel. Chunks already on disk are skipped, which
# makes the script resumable.
#
# Usage:
#   bash scripts/fetch_malecns.sh [VERSION] [DEST_DIR]
set -euo pipefail

VERSION="${1:-v1.0}"
DEST="${2:-/root/autodl-tmp/data/malecns/${VERSION}}"
BASE="https://storage.googleapis.com/flyem-male-cns/${VERSION}/connectome-data/flat-connectome"
CHUNK="${CHUNK_SIZE:-8388608}"
WORKERS="${WORKERS:-16}"

# Set ONLY=weights|annotations to fetch a single file (useful when resuming).
if [ "${ONLY:-all}" = "weights" ]; then
  FILES=("connectome-weights-male-cns-${VERSION}-minconf-0.5.feather")
elif [ "${ONLY:-all}" = "annotations" ]; then
  FILES=("body-annotations-male-cns-${VERSION}-minconf-0.5.feather")
else
  FILES=(
    "body-annotations-male-cns-${VERSION}-minconf-0.5.feather"
    "connectome-weights-male-cns-${VERSION}-minconf-0.5.feather"
  )
fi

mkdir -p "$DEST"

fetch_one() {
  local name="$1"
  local url="$BASE/$name"
  local out="$DEST/$name"
  local parts="$DEST/parts_$name"
  mkdir -p "$parts"

  local total
  total=$(curl -sSI "$url" | grep -i '^content-length' | tr -d '\r' | awk '{print $2}')
  local chunks=$(( (total + CHUNK - 1) / CHUNK ))
  echo "[$name] total=$total chunks=$chunks chunk=$CHUNK workers=$WORKERS"

  export url parts CHUNK total
  seq 0 $((chunks - 1)) | xargs -P "$WORKERS" -I{} bash -c '
    i={}
    start=$(( i * CHUNK ))
    end=$(( start + CHUNK - 1 ))
    [ "$end" -gt $(( total - 1 )) ] && end=$(( total - 1 ))
    out="$parts/part_$(printf %06d "$i")"
    if [ -s "$out" ]; then exit 0; fi
    for attempt in 1 2 3 4 5 6; do
      if curl -sS --range "${start}-${end}" -o "$out" "$url"; then break; fi
      sleep 3
    done
    [ -s "$out" ] || echo "FAILED chunk $i" >&2
  '

  echo "[$name] assembling"
  cat "$parts"/part_* > "$out"

  local size
  size=$(stat -c%s "$out")
  if [ "$size" = "$total" ]; then
    echo "[$name] OK size=$size"
    rm -rf "$parts"
  else
    echo "[$name] SIZE MISMATCH size=$size expected=$total" >&2
    return 1
  fi
}

for f in "${FILES[@]}"; do
  fetch_one "$f"
done

cd "$DEST" && sha256sum *.feather > SHA256SUMS.txt
echo "DONE -> $DEST"
