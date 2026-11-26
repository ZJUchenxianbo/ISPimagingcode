#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--force] [bem|schur|water ...]"
    echo "With no targets, build the three forward CUDA libraries in the GPU directory."
    echo "Set CUDA_ROOT and CUDSS_ROOT to your development installations."
    echo "Set PLANAR_CUDA_ARCH for another GPU architecture (required for your target GPU) and use --force after changing toolchains."
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
force=0
if [[ ${1:-} == --force ]]; then
    force=1
    shift
fi
if (( $# == 0 )); then
    set -- bem schur water
fi
for target in "$@"; do
    case "$target" in
        bem|schur|water) ;;
        *) echo "Unknown CUDA build target: $target" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ $(uname -s) != Linux ]]; then
    echo "The CUDA libraries require Linux. Use main.py -forward --backend cpu for CPU runs." >&2
    exit 1
fi

cuda_root=${PLANAR_INVERSION_CUDA_ROOT:-${CUDA_ROOT:-/usr/local/cuda}}
driver_root=${PLANAR_INVERSION_DRIVER_ROOT:-}
cudss_default=${driver_root:+$driver_root/cudss}
cudss_root=${PLANAR_INVERSION_CUDSS_ROOT:-${CUDSS_ROOT:-${cudss_default:-/opt/nvidia/cudss}}}
nvcc="$cuda_root/bin/nvcc"
cuda_lib="$cuda_root/targets/x86_64-linux/lib"
if [[ ! -d "$cuda_lib" ]]; then cuda_lib="$cuda_root/lib64"; fi
cudss_lib="$cudss_root/lib"
if [[ ! -d "$cudss_lib" ]]; then cudss_lib="$cudss_root/lib64"; fi

cd -- "$(dirname -- "$0")"
if ! command -v flock >/dev/null; then
    echo "The GPU build requires flock from util-linux." >&2
    exit 1
fi
exec 9>.gpu-build.lock
flock 9

: "${PLANAR_CUDA_ARCH:?Set PLANAR_CUDA_ARCH to the target architecture, for example sm_80}"

common=(-O3 -std=c++17 "-arch=${PLANAR_CUDA_ARCH}" --shared -Xcompiler -fPIC)

build() (
    local library=$1 source=$2
    shift 2
    if (( ! force )) && [[ -s "$library" && ! "$source" -nt "$library" && ! build.sh -nt "$library" ]]; then
        if [[ "$source" != panel_schur.cpp || ! panel_assembly.cu -nt "$library" ]]; then
            return
        fi
    fi
    if [[ ! -x "$nvcc" ]]; then
        echo "Missing nvcc at $nvcc. Set CUDA_ROOT to a CUDA Toolkit installation." >&2
        return 1
    fi
    if [[ "$library" == libgpu_panel_schur.panel-normal.so && ! -f "$cudss_root/include/cudss.h" ]]; then
        echo "Missing $cudss_root/include/cudss.h. Set CUDSS_ROOT to a cuDSS development installation." >&2
        return 1
    fi
    local temporary
    temporary=$(mktemp "${library}.XXXXXXXX")
    trap 'rm -f -- "$temporary"' EXIT
    printf 'Building %s\n' "$library"
    "$nvcc" "${common[@]}" "$@" "$source" -o "$temporary"
    mv -- "$temporary" "$library"
)

for target in "$@"; do
    case "$target" in
        bem) build libplanar_bem_cuda_v7b.so bem_kernels.cu ;;
        schur)
            build libgpu_panel_schur.panel-normal.so panel_schur.cpp \
                --fmad=false -DPANEL_DEVICE_ASSEMBLY panel_assembly.cu \
                -I"$cuda_root/include" -I"$cudss_root/include" -L"$cuda_lib" -L"$cudss_lib" \
                -lcudart -lcudss -lcusparse -lcublas
            ;;
        water)
            build libplanar_water_tangent.so water.cu \
                -L"$cuda_lib" -lcublas -lcusolver -lcudart
            ;;
    esac
done
