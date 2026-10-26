#!/usr/bin/env bash
# Run on the host with the user's configured bcecmd. Safe to restart.
set -euo pipefail
DATASET_DIR=${1:?Usage: fetch_packed_data.sh DATASET_DIR PYTHON BOS_SOURCE}
PYTHON_BIN=${2:?Python executable required}
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PACKED_ROOT="$DATASET_DIR/flood-g1-packed-v1"
ARCHIVE="$DATASET_DIR/flood-g1-packed-v1.tar.gz"
SOURCE=${3:?BOS archive URI required}
mkdir -p "$DATASET_DIR"
exec 9>"$DATASET_DIR/flood-g1-download.lock"
flock -n 9
if [[ -f "$PACKED_ROOT/VERIFIED.json" ]]; then
    exit 0
fi
if [[ ! -f "$ARCHIVE" ]]; then
    ready=0
    for ((attempt=0; attempt<288; attempt++)); do
        listing=$(bcecmd bos ls "$SOURCE" 2>/dev/null || true)
        if [[ "$listing" == *flood-g1-packed-v1.tar.gz* ]]; then
            ready=1
            break
        fi
        sleep 300
    done
    [[ "$ready" == 1 ]] || { echo 'Timed out waiting for feature-bank upload' >&2; exit 1; }
    bcecmd bos cp "$SOURCE" "$ARCHIVE.part" --disable-bar
    mv "$ARCHIVE.part" "$ARCHIVE"
fi
if [[ ! -f "$PACKED_ROOT/manifest.json" ]]; then
    STAGING_DIR=$(mktemp -d "$DATASET_DIR/.flood-g1-extract.XXXXXX")
    tar -xzf "$ARCHIVE" -C "$STAGING_DIR"
    [[ ! -e "$PACKED_ROOT" ]] || { echo 'Unexpected incomplete destination; inspect before retry' >&2; exit 1; }
    mv "$STAGING_DIR/flood-g1-packed-v1" "$PACKED_ROOT"
    rmdir "$STAGING_DIR"
fi
"$PYTHON_BIN" "$REPO_DIR/scripts/g1/verify_data.py" "$PACKED_ROOT" > "$PACKED_ROOT/VERIFIED.json.tmp"
mv "$PACKED_ROOT/VERIFIED.json.tmp" "$PACKED_ROOT/VERIFIED.json"
