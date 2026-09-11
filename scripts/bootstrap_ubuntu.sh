#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/vendor"

[[ "$(uname -s)" == "Linux" ]] || { echo "Linux veya WSL2 gerekir." >&2; exit 2; }
for cmd in git curl; do command -v "$cmd" >/dev/null || { echo "Eksik araç: $cmd" >&2; exit 2; }; done

if ! command -v uv >/dev/null; then
  installer="$(mktemp)"
  curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/install.sh -o "$installer"
  sh "$installer"
  rm -f "$installer"
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null || { echo "uv bulunamadı." >&2; exit 2; }
mkdir -p "$VENDOR"

clone_at() {
  local url="$1" destination="$2" commit="$3"
  [[ -d "$destination/.git" ]] || git clone --filter=blob:none "$url" "$destination"
  git -C "$destination" fetch --depth 1 origin "$commit"
  git -C "$destination" checkout --detach "$commit"
  test "$(git -C "$destination" rev-parse HEAD)" = "$commit"
}

clone_at https://github.com/GaTech-RL2/ot-sim2real.git "$VENDOR/ot-sim2real" 114704b6b381b410cc14a51cb16952d1f4e8c69d
clone_at https://github.com/ARISE-Initiative/robosuite.git "$VENDOR/robosuite" b9d8d3de5e3dfd1724f4a0e6555246c460407daa
clone_at https://github.com/NVlabs/mimicgen.git "$VENDOR/mimicgen" 72bd767c255545f462e7ccfb2731f2e5d4c1d9bb
clone_at https://github.com/touristCheng/model-zoo-mugs-only.git "$VENDOR/model-zoo-mugs-only" fd9f480c021e0681579724741ed4755e492f5426

if git -C "$VENDOR/ot-sim2real" apply --check "$ROOT/patches/0001-demo-rights-and-seeds.patch" 2>/dev/null; then
  git -C "$VENDOR/ot-sim2real" apply "$ROOT/patches/0001-demo-rights-and-seeds.patch"
elif ! git -C "$VENDOR/ot-sim2real" apply --reverse --check "$ROOT/patches/0001-demo-rights-and-seeds.patch" 2>/dev/null; then
  echo "Kayıtlı patch temiz uygulanamadı." >&2; exit 3
fi

[[ -x "$ROOT/.venv/bin/python" ]] || uv venv --python 3.10 "$ROOT/.venv"
source "$ROOT/.venv/bin/activate"
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1
uv pip install -r "$ROOT/environment/requirements.txt"
uv pip install --no-deps -e "$VENDOR/robosuite" -e "$VENDOR/mimicgen" -e "$VENDOR/model-zoo-mugs-only" -e "$VENDOR/ot-sim2real"
python "$ROOT/scripts/self_check.py" --runtime
echo "Kurulum tamamlandı: $ROOT/.venv"

