#!/usr/bin/env bash
# MuJoCo GUI/offscreen backend 조합을 자동으로 실험하고 성공/실패를 로그로 남긴다.
#
# 이 스크립트는 scripts/debug_mujoco_viewer.py(같은 디렉터리)를 여러 환경변수 조합으로
# 실행해 본다. GUI 조합은 단순히 "예외 없이 창이 떴는가"만 보지 않고, X11(XWD)로 창을
# 실제로 캡처해서 픽셀 표준편차로 "내용이 실제로 그려졌는가"까지 자동으로 판정한다
# (xwd/xwininfo가 있고 이 셸이 GUI와 같은 DISPLAY에 접근 가능할 때만 - 그렇지 않으면
# "미확인"으로 남긴다. 지원되지 않는 backend를 성공했다고 주장하지 않는다).
#
# offscreen 조합(MUJOCO_GL=egl/osmesa)은 mujoco.Renderer 기준이며, GUI 창(GLFW)과는
# 무관하다 - GUI 창은 항상 GLFW를 직접 사용하고 MUJOCO_GL의 영향을 받지 않는다. 이 스크립트는
# 그 사실도 로그에 명시한다.
#
# 사용법:
#   scripts/run_mujoco_gui_diagnostics.sh [python-interpreter] [로그-디렉터리]
#
# 예:
#   scripts/run_mujoco_gui_diagnostics.sh ~/lerobot/.venv/bin/python reports/mujoco_gui_debug

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEBUG_SCRIPT="$SCRIPT_DIR/debug_mujoco_viewer.py"

LOG_DIR="${2:-$PROJECT_ROOT/reports/mujoco_gui_debug}"
mkdir -p "$LOG_DIR"
SUMMARY_LOG="$LOG_DIR/backend_diagnostics_summary.log"
: > "$SUMMARY_LOG"

BAR="===================================================================="

log() {
    echo "$1" | tee -a "$SUMMARY_LOG"
}

# ---- python 인터프리터 결정 --------------------------------------------
PYBIN="${1:-}"
if [ -z "$PYBIN" ]; then
    for candidate in python3 "$HOME/lerobot/.venv/bin/python"; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "import mujoco" >/dev/null 2>&1; then
            PYBIN="$candidate"
            break
        fi
    done
fi
if [ -z "$PYBIN" ] || ! "$PYBIN" -c "import mujoco" >/dev/null 2>&1; then
    log "[오류] mujoco가 import되는 python 인터프리터를 찾지 못했습니다. 첫 번째 인자로 직접 지정하세요."
    exit 1
fi

log "$BAR"
log "[진단] MuJoCo GUI/offscreen backend 조합 실험"
log "$BAR"
log "[환경] python = $PYBIN ($($PYBIN -c 'import mujoco; print(mujoco.__version__)' 2>/dev/null))"
log "[환경] DISPLAY=${DISPLAY:-<미설정>} WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-<미설정>}"
log "[환경] 로그 디렉터리 = $LOG_DIR"
log "$BAR"

HAVE_X11_TOOLS=1
if ! command -v xwininfo >/dev/null 2>&1 || ! command -v xwd >/dev/null 2>&1; then
    HAVE_X11_TOOLS=0
    log "[경고] xwininfo/xwd가 없어 GUI 창 내용을 자동으로 캡처/검증할 수 없습니다. 창 생성 여부(예외 발생 여부)만 판정합니다."
fi

# ---- GUI(glfw 창) 캡처 후 픽셀 분산으로 실제 렌더링 여부 판정 ----------
capture_and_check() {
    # $1 = 결과를 저장할 변수명 접두사가 아니라, stdout으로 "판정문구|std편차" 출력
    local wait_s="$1"
    sleep "$wait_s"
    local wid
    wid=$(xwininfo -root -tree 2>/dev/null | grep -oP '0x[0-9a-f]+(?=\s+"MuJoCo)' | head -1)
    if [ -z "$wid" ]; then
        echo "창을 찾지 못함|-"
        return
    fi
    local xwd_path="$LOG_DIR/_capture_$$.xwd"
    if ! xwd -id "$wid" -out "$xwd_path" 2>/dev/null; then
        echo "xwd 캡처 실패(창이 이미 닫혔을 수 있음)|-"
        return
    fi
    local result
    result=$("$PYBIN" - "$xwd_path" <<'PYEOF'
import struct, sys
import numpy as np

path = sys.argv[1]
try:
    with open(path, "rb") as f:
        data = f.read()
    header = struct.unpack(">25I", data[:100])
    (header_size, _fv, _pf, _pd, width, height, _xoff, _bo, _bu, _bbo, _bpad,
     bits_per_pixel, bytes_per_line, _vc, _rm, _gm, _bm, _brgb,
     _cme, ncolors, _ww, _wh, _wx, _wy, _wb) = header
    offset = header_size + ncolors * 12
    if bits_per_pixel != 32:
        print(f"미지원 bpp={bits_per_pixel}|-")
        sys.exit(0)
    pixel_data = data[offset: offset + bytes_per_line * height]
    arr = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, bytes_per_line)[:, : width * 4]
    std = float(arr.std())
    print(f"ok|{std:.2f}")
except Exception as exc:
    print(f"파싱 실패({exc})|-")
PYEOF
)
    rm -f "$xwd_path"
    echo "$result"
}

run_gui_combo() {
    local label="$1"; shift
    log ""
    log "[검사] GUI 조합: $label"
    log "  env: $*"
    local out_log="$LOG_DIR/gui_${label// /_}.log"

    env "$@" "$PYBIN" "$DEBUG_SCRIPT" --mode passive --duration 6 > "$out_log" 2>&1 &
    local pid=$!

    local verdict="미확인"
    if [ "$HAVE_X11_TOOLS" -eq 1 ]; then
        local capture
        capture=$(capture_and_check 3)
        local status="${capture%%|*}"
        local std="${capture##*|}"
        if [ "$status" = "ok" ]; then
            # std가 충분히 크면(단색이 아니면) 실제 콘텐츠가 그려진 것으로 판정한다.
            if (( $(echo "$std > 5.0" | bc -l 2>/dev/null || echo 0) )); then
                verdict="PASS (픽셀 표준편차=$std, 단색 아님 -> 실제 콘텐츠 렌더링 확인)"
            else
                verdict="FAIL (픽셀 표준편차=$std, 거의 단색 -> 창은 떴지만 내용 없음/블랭크 가능성)"
            fi
        else
            verdict="미확인 ($status)"
        fi
    fi

    wait "$pid" 2>/dev/null
    local exit_code=$?

    local loop_completed=0
    if grep -q "\[통과\] viewer 정상 종료" "$out_log" 2>/dev/null; then
        loop_completed=1
    fi

    if [ "$exit_code" -eq 139 ] && [ "$loop_completed" -eq 1 ]; then
        log "  [결과] step/sync 루프는 정상 완료했지만 프로세스 종료(정리) 중 SIGSEGV 발생 (exit=139) - $out_log 확인"
        log "  [판정] $verdict (단, 프로세스 종료 시 native 크래시 있음 - Mesa llvmpipe 관련 가능성, 아래 보고서 참고)"
    elif [ "$exit_code" -ne 0 ] && [ "$exit_code" -ne 130 ]; then
        log "  [결과] 프로세스 예외 종료 (exit=$exit_code) - $out_log 확인"
        log "  [판정] FAIL (실행 자체가 실패함)"
    else
        log "  [결과] 프로세스 정상 종료 (exit=$exit_code)"
        log "  [판정] $verdict"
    fi
}

run_offscreen_combo() {
    local label="$1"; shift
    log ""
    log "[검사] offscreen 조합: $label (mujoco.Renderer, GUI 창과 무관)"
    log "  env: $*"
    local out_log="$LOG_DIR/offscreen_${label// /_}.log"
    if env "$@" "$PYBIN" "$DEBUG_SCRIPT" --mode render-offscreen --frames 3 \
        --output-dir "$LOG_DIR/offscreen_${label// /_}" > "$out_log" 2>&1; then
        log "  [판정] PASS (offscreen 렌더링 성공, PNG 저장됨) - $out_log"
    else
        log "  [판정] FAIL 또는 미지원 (이 backend는 이 환경에서 동작하지 않음) - $out_log 확인"
    fi
}

# ---- offscreen 조합 (mujoco.Renderer 기준) ------------------------------
log "$BAR"
log "[구간] offscreen 렌더링 backend 비교 (mujoco.Renderer 전용, GUI와 무관)"
log "$BAR"
run_offscreen_combo "default"
run_offscreen_combo "MUJOCO_GL_egl" MUJOCO_GL=egl
run_offscreen_combo "MUJOCO_GL_osmesa" MUJOCO_GL=osmesa
run_offscreen_combo "LIBGL_SOFTWARE" LIBGL_ALWAYS_SOFTWARE=1

# ---- GUI 조합 (GLFW 창 기준) --------------------------------------------
log ""
log "$BAR"
log "[구간] GUI(GLFW 창) backend/환경변수 비교"
log "[참고] MUJOCO_GL은 GLFW 창 생성에 영향을 주지 않는다 (GUI는 항상 GLFW 직접 사용)."
log "$BAR"
run_gui_combo "default"
run_gui_combo "LIBGL_SOFTWARE" LIBGL_ALWAYS_SOFTWARE=1
if [ -n "${WAYLAND_DISPLAY:-}" ]; then
    run_gui_combo "no_wayland_force_x11" WAYLAND_DISPLAY=
    run_gui_combo "no_wayland_and_software" WAYLAND_DISPLAY= LIBGL_ALWAYS_SOFTWARE=1
else
    log ""
    log "[안내] WAYLAND_DISPLAY가 이미 설정되어 있지 않아 X11 강제 조합은 건너뜁니다."
fi

log ""
log "$BAR"
log "[요약] 전체 로그: $SUMMARY_LOG"
log "[요약] 개별 로그/PNG: $LOG_DIR"
log "$BAR"
