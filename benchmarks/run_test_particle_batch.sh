#!/usr/bin/env bash
set -euo pipefail

# Override sweep with: KN_VALUES="50 100" ./run_test_particle_batch.sh
if [[ -n "${KN_VALUES:-}" ]]; then
  read -ra VALUES <<<"${KN_VALUES}"
else
  VALUES=(50 100 500 1000 5000 10000 50000 100000)
fi

TARGET=bench_particle

BENCHMARKS=(
  BM_OptIndex_Create
  BM_BMI_Create
  BM_PoolBMI_Create
  BM_PartOptIndex_Create
  BM_OptIndex_FindPrimary
  BM_BMI_FindPrimary
  BM_PoolBMI_FindPrimary
  BM_PartOptIndex_FindPrimary
  BM_OptIndex_Remove
  BM_BMI_Remove
  BM_PoolBMI_Remove
  BM_PartOptIndex_Remove
  BM_OptIndex_BulkIterate
  BM_BMI_BulkIterate
  BM_PoolBMI_BulkIterate
  BM_PartOptIndex_BulkIterate
  BM_OptIndex_OrderedIterate
  BM_BMI_OrderedIterate
  BM_PoolBMI_OrderedIterate
  BM_PartOptIndex_OrderedIterate
  BM_OptIndex_MassRange
  BM_BMI_MassRange
  BM_PoolBMI_MassRange
  BM_PartOptIndex_MassRange
  BM_OptIndex_Modify
  BM_BMI_Modify
  BM_PoolBMI_Modify
  BM_PartOptIndex_Modify
  BM_OptIndex_ModifyX
  BM_BMI_ModifyX
  BM_PoolBMI_ModifyX
  BM_PartOptIndex_ModifyX
  BM_OptIndex_LevelWalk
  BM_BMI_LevelWalk
  BM_PoolBMI_LevelWalk
  BM_PartOptIndex_LevelWalk
  BM_OptIndex_MixedSteadyState
  BM_BMI_MixedSteadyState
  BM_PoolBMI_MixedSteadyState
  BM_PartOptIndex_MixedSteadyState
)

RUN_MODE="${1:-default}"
BENCH_CPU="${BENCH_CPU:-0}"
DEBUG="${DEBUG:-0}"

COMMON_BENCH_ARGS=(
  --benchmark_repetitions=200
  --benchmark_out_format=json
)

run_cmd() {
  local exe="$1"
  shift

  case "$RUN_MODE" in
    default)
      "$exe" "$@"
      ;;
    pinned)
      taskset -c "${BENCH_CPU}" "$exe" "$@"
      ;;
    background)
      taskpolicy -b "$exe" "$@"
      ;;
    *)
      echo "Unknown RUN_MODE: $RUN_MODE"
      echo "Use: default, pinned, or background"
      exit 1
      ;;
  esac
}

find_exe() {
  local build_dir="$1"

  local candidates=(
    "${build_dir}/${TARGET}"
    "${build_dir}/src/${TARGET}"
    "${build_dir}/bench/${TARGET}"
    "${build_dir}/benchmark/${TARGET}"
    "${build_dir}/Debug/${TARGET}"
    "${build_dir}/Release/${TARGET}"
  )

  for p in "${candidates[@]}"; do
    if [[ -x "$p" ]]; then
      echo "$p"
      return 0
    fi
  done

  return 1
}

for kn in "${VALUES[@]}"; do
  for rev_index in false true; do
    suffix=""
    if [[ "${rev_index}" == "true" ]]; then
      suffix="_rev"
    fi

    build_dir="build_kn_${kn}${suffix}"
    data_dir="data_kn_${kn}${suffix}"

    # rm -rf "${data_dir}"
    mkdir -p "${data_dir}"

    echo "============================================================"
    echo "Building ${TARGET} with SET_KN=${kn} REV_INDEX=${rev_index}"
    echo "============================================================"

    cmake -S . -B "${build_dir}" -DSET_KN_VALUE="${kn}" -DREV_INDEX="${rev_index}" -DWITH_HASH=true -DCMAKE_BUILD_TYPE=Release
    cmake --build "${build_dir}" --target "${TARGET}" -j

    exe="$(find_exe "${build_dir}")" || {
      echo "Could not find executable ${TARGET} under ${build_dir}"
      exit 1
    }

    echo "Using executable: ${exe}"

    for bench in "${BENCHMARKS[@]}"; do
      out_json="${data_dir}/result_${kn}${suffix}_${bench}.json"
      out_log="${data_dir}/result_${kn}${suffix}_${bench}.log"

      echo
      echo "--- Running ${bench} with SET_KN=${kn} REV_INDEX=${rev_index} ---"

      run_cmd "${exe}" \
        "${COMMON_BENCH_ARGS[@]}" \
        "--benchmark_filter=^${bench}$" \
        "--benchmark_out=${out_json}" \
        "--benchmark_min_time=0.1s" \
        2>&1 | tee "${out_log}"
    done

    if [[ "${DEBUG}" == "0" ]]; then
      rm -rf "${build_dir}"
    else
      echo "DEBUG=${DEBUG}: keeping ${build_dir}"
    fi
  done
done
