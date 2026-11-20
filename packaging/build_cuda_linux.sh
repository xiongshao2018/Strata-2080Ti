#!/bin/sh
# Build the Linux CUDA engine package for two RTX 2080 Ti cards (Turing, compute capability 7.5) over NVLink.
#
#   VARIANT=cuda13 packaging/build_cuda_linux.sh     # CUDA 13, machine code for sm_75 only
#
# Run it on a Linux x86_64 box with the CUDA toolkit of that variant on PATH (nvcc), a Python 3 venv with requirements.txt and
# requirements-build.txt, and the pinned llama.cpp in third_party/llama.cpp (python -c "import setup; setup.get_llama_cpp()").
# The package lands in $OUT (default ./out-engine-<variant>). Nothing is uploaded.
#
# This fork targets one hardware pair, so the architecture list is not a variable any more: 75 only. The cuda12 variant
# (Pascal / Volta, sm_60 / 61 / 70) is refused here rather than built with code this fork never runs.
set -eu
variant=${VARIANT:-cuda13}
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
case "$variant" in
  cuda13) archs=75; mindriver=580.65; sm60=OFF ;;
  *) echo "build_cuda_linux.sh: this fork builds the cuda13 (sm_75) engine only; VARIANT=$variant is not supported" >&2; exit 2 ;;
esac
out=${OUT:-$repo/out-engine-$variant}
build=${BUILD:-$repo/build-$variant}
py=${PYTHON:-python3}
cudafull=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
case "$variant:$cudafull" in
  cuda12:12.*|cuda13:13.*) ;;
  *) echo "build_cuda_linux.sh: $variant needs the matching CUDA toolkit; nvcc here is ${cudafull:-missing}" >&2; exit 2 ;;
esac

real=""; for a in $(echo "$archs" | tr ';,' '  '); do real="$real;${a}-real"; done; real=${real#;}
last=$(echo "$archs" | tr ';,' '  ' | awk '{print $NF}')
cd "$repo"
cmake -G Ninja -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
    -DSTRATA_EXPERIMENTAL_SM60=$sm60 -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
    "-DCMAKE_CUDA_ARCHITECTURES=${real};${last}-virtual" -DSTRATA_GGML_DIR="$repo/third_party/llama.cpp"
cmake --build "$build" --target strata strata_kq -j "$(nproc)"

ver=$(sed -n 's/^[[:space:]]*project([^)]*VERSION[[:space:]]*\([0-9][0-9.]*\).*/\1/p' CMakeLists.txt | head -1)
stage=$(mktemp -d "${TMPDIR:-/tmp}/strata-engine-stage.XXXXXX")
trap 'rm -rf "$stage"' EXIT
cp "$build/strata" "$stage/"
[ -f "$build/libstrata_kq.so" ] && cp "$build/libstrata_kq.so" "$stage/"
echo "{\"version\": \"$ver\", \"source\": \"packaging/build_cuda_linux.sh\", \"archs\": \"$archs\", \"cuda\": \"$cudafull\", \"portable\": true}" > "$stage/BUILD.json"
"$py" packaging/collect_libs.py --binary "$build/strata" --search /usr/local/cuda/lib64 --out "$stage" --also 'libcublasLt.so*'
arch_args=""; for a in $(echo "$archs" | tr ';,' '  '); do arch_args="$arch_args --arch sm_$a"; done
mkdir -p "$out"
# shellcheck disable=SC2086
"$py" packaging/package_engine.py --src "$stage" --backend cuda --version "$ver" --out "$out" $arch_args --arch "compute_$last" \
    --min-driver nvidia=$mindriver --variant "$variant" --platform linux-x86_64 --build-id "release-${ver}-${variant}-${cudafull}" \
    --include SYSTEM-LIBS.txt --include 'libstrata_kq*'
for d in "$out"/*/; do "$py" packaging/package_engine.py --verify "$d"; "$py" packaging/binary_deps.py "$d"; done
echo "built: $out"
