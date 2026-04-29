#!/usr/bin/env bash
#
# heterosplat — run the dev container.
#
# Wraps `docker run` with the same flags docker_builder produces, but reads
# per-machine config (GPU index, image tag) from
# Scripts/run_configuration.yml. The mount source is auto-detected from this
# script's location, so the same checked-in script works across machines
# regardless of where the heterosplat repo lives on disk.
#
# Usage:
#   ./Scripts/run_container.sh                   # interactive shell in /heterosplat
#   ./Scripts/run_container.sh '<command>'       # run one command, exit
#
# Examples:
#   ./Scripts/run_container.sh
#   ./Scripts/run_container.sh 'cmake -S CUDA/Heterosplat/Source -B build'
#   ./Scripts/run_container.sh 'cmake --build build -j6 && ./build/Check'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_FILE="${SCRIPT_DIR}/run_configuration.yml"
EXAMPLE_FILE="${SCRIPT_DIR}/run_configuration.yml.example"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Error: ${CONFIG_FILE} not found." >&2
  echo >&2
  echo "Copy the template and edit gpu_id for this machine:" >&2
  echo "  cp ${EXAMPLE_FILE} ${CONFIG_FILE}" >&2
  exit 1
fi

# Tiny YAML parser. Only handles flat `key: value` lines; comments after the
# value are stripped. Returns empty string when key is absent (no error so the
# caller can decide whether the key is required).
read_yaml_key() {
  local key="$1"
  awk -v key="${key}" '
    {
      pos = index($0, ":")
      if (pos == 0) next
      k = substr($0, 1, pos - 1)
      v = substr($0, pos + 1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", k)
      if (k != key) next
      sub(/[[:space:]]*#.*$/, "", v)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
      print v
      exit
    }
  ' "${CONFIG_FILE}"
}

GPU_ID="$(read_yaml_key gpu_id)"
IMAGE="$(read_yaml_key image)"

if [[ -z "${GPU_ID}" ]]; then
  echo "Error: gpu_id is missing or empty in ${CONFIG_FILE}" >&2
  exit 1
fi
if [[ -z "${IMAGE}" ]]; then
  IMAGE="heterosplat:26.02-py3"
fi

# docker_builder mounts the repo at /heterosplat. Match that so the same
# CMake build dir works regardless of how the container was launched.
MOUNT_DST="/heterosplat"

DOCKER_BASE_ARGS=(
  --rm
  --gpus "device=${GPU_ID}"
  -e NVIDIA_DISABLE_REQUIRE=1
  -e CUDA_VISIBLE_DEVICES=0
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  -v "${REPO_ROOT}:${MOUNT_DST}"
  -w "${MOUNT_DST}"
)

if [[ $# -eq 0 ]]; then
  TTY_ARGS=(-it)
  SHELL_ARGS=()
  MODE_DESC="interactive shell"
else
  TTY_ARGS=()
  SHELL_ARGS=(-c "$*")
  MODE_DESC="command: $*"
fi

CMD=(
  docker run
  "${TTY_ARGS[@]}"
  "${DOCKER_BASE_ARGS[@]}"
  "${IMAGE}"
  bash
  "${SHELL_ARGS[@]}"
)

echo "==> heterosplat container"
echo "    Image:  ${IMAGE}"
echo "    GPU:    device=${GPU_ID}  (CUDA_VISIBLE_DEVICES=0 inside)"
echo "    Mount:  ${REPO_ROOT}  ->  ${MOUNT_DST}"
echo "    Mode:   ${MODE_DESC}"
echo
echo "==> docker run command:"
printf '    '
printf '%q ' "${CMD[@]}"
printf '\n\n'

exec "${CMD[@]}"
