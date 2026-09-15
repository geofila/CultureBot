#!/usr/bin/env bash
#
# Clean-rebuild helper for the SearchCultureBot stack.
#
#   ./rebuild.sh           Rebuild the pipeline code layers and recreate both containers.
#                          Cached pip/PyTorch layers are reused, so this takes ~1 minute.
#                          This is what fixes "my code change didn't take effect".
#
#   ./rebuild.sh --full    Rebuild every layer from the base image (docker build --no-cache)
#                          and re-pull the Open WebUI image. Re-downloads the CUDA PyTorch
#                          wheels, ~3 GB, 15-30 minutes. Literally from the beginning.
#
#   ./rebuild.sh --wipe    Also delete Open WebUI's data: accounts, chats and the pipelines
#                          connection. Asks for confirmation. Combine with --full if you want.
#
# The FAISS/BM25 index cache (data/pipelines-cache) is always removed, so the index is
# rebuilt from your dataset on the first question after a rebuild.
#
set -euo pipefail
cd "$(dirname "$0")"

FULL=0
WIPE=0
for arg in "$@"; do
  case "$arg" in
    --full) FULL=1 ;;
    --wipe) WIPE=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

# Docker creates these directories as root, so a plain rm may fail. Fall back to deleting
# them from inside a throwaway container, which runs as root.
remove_dir() {
  local target="$1"
  [ -e "$target" ] || return 0
  if rm -rf "$target" 2>/dev/null; then
    echo "  removed $target"
  else
    docker run --rm -v "$PWD/data:/d" --entrypoint rm busybox -rf "/d/$(basename "$target")"
    echo "  removed $target (via container)"
  fi
}

echo "==> Stopping and removing containers"
docker compose down

echo "==> Removing the search-index cache"
remove_dir "$PWD/data/pipelines-cache"

if [ "$WIPE" = 1 ]; then
  echo
  echo "!!  --wipe will delete data/open-webui: every account, every chat, and the"
  echo "!!  connection to the pipelines server. This cannot be undone."
  printf "    Type 'yes' to continue: "
  read -r answer
  if [ "$answer" = "yes" ]; then
    remove_dir "$PWD/data/open-webui"
  else
    echo "  skipped — Open WebUI data kept"
  fi
fi

if [ "$FULL" = 1 ]; then
  echo "==> Rebuilding the pipelines image from scratch (no cache) — this takes a while"
  docker compose build --no-cache --pull pipelines
  echo "==> Re-pulling the Open WebUI image"
  docker compose pull open-webui
else
  echo "==> Rebuilding the pipelines image (cached base layers, current code)"
  docker compose build pipelines
fi

echo "==> Starting the stack"
docker compose up -d --force-recreate

echo
docker compose ps
echo
echo "The dataset scan appears in the log a few seconds after startup:"
echo "  docker compose logs pipelines | grep -A20 'Dataset scan'"
