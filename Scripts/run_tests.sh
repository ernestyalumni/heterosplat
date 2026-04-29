#!/usr/bin/env bash
#
# heterosplat — run the unit-test binary on the host (no container).
#
# Reads gpu_id from Scripts/run_configuration.yml and exports
# CUDA_VISIBLE_DEVICES so kernel launches go to the chosen GPU. Use this for
# rapid host iteration; the docker container is reserved for the gsplat-Python
# numerical-correctness oracle (Phase 0b) and deploy-shape smoke checks.
#
# Usage:
#   ./Scripts/run_tests.sh                                # run all tests
#   ./Scripts/run_tests.sh --gtest_filter=Tensor.*        # gtest filter
#   ./Scripts/run_tests.sh --gtest_list_tests             # list discovered tests
#   HETEROSPLAT_BUILD_DIR=build ./Scripts/run_tests.sh    # override build path
#
# Manual one-liner equivalent (skip this script entirely; substitute your own
# gpu_id and build dir):
#
#   CUDA_VISIBLE_DEVICES=1 ./CUDA/Heterosplat/Build/Check
#
# All extra arguments are forwarded verbatim to the Check binary.

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

# Same flat-YAML parser shape as run_container.sh — handles `key: value`,
# strips trailing comments, returns empty string on missing key.
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

if [[ -z "${GPU_ID}" ]]; then
  echo "Error: gpu_id is missing or empty in ${CONFIG_FILE}" >&2
  exit 1
fi

BUILD_DIR="${HETEROSPLAT_BUILD_DIR:-${REPO_ROOT}/CUDA/Heterosplat/Build}"
CHECK_BIN="${BUILD_DIR}/Check"

if [[ ! -x "${CHECK_BIN}" ]]; then
  echo "Error: ${CHECK_BIN} not found or not executable." >&2
  echo >&2
  echo "Build first (host):" >&2
  echo "  cmake -S CUDA/Heterosplat/Source -B ${BUILD_DIR} && cmake --build ${BUILD_DIR} -j6" >&2
  exit 1
fi

echo "==> heterosplat tests (host)"
echo "    GPU:    CUDA_VISIBLE_DEVICES=${GPU_ID}"
echo "    Binary: ${CHECK_BIN}"
if [[ $# -gt 0 ]]; then
  echo "    Args:   $*"
fi
echo

# Force PCI-bus indexing so `gpu_id` means the same thing on host
# (this script) and inside docker (`docker run --gpus device=$gpu_id`).
# By default, CUDA orders devices by compute capability, which can flip
# the index relative to PCI / docker / `nvidia-smi -L`. Setting
# CUDA_DEVICE_ORDER=PCI_BUS_ID aligns them.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
exec "${CHECK_BIN}" "$@"
