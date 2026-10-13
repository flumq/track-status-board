import base64
import io

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, binary_erosion

# ─────────────────────────────────────────────
# 1. 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="테스트 트랙 코스 현황판",
    page_icon="🏁",
    layout="wide",
)

st.title("🏁 테스트 트랙 팀-코스 매칭 실시간 현황판")

# ─────────────────────────────────────────────
# 2. 팀 리스트 및 팀별 지정 색상 (RGB)
# ─────────────────────────────────────────────
team_colors = {
    "Team 1": (0, 102, 204),        # 파란색
    "Team 2": (220, 50, 50),        # 빨간색
    "Team 3": (0, 188, 212),        # 청록색
    "Team 4": (46, 139, 87),        # 초록색
    "Team 5": (138, 43, 226),       # 보라색
    "공사 및 작업": (255, 110, 0),   # 비비드 오렌지
}

course_files = {
    "A코스": "A코스-8.png",
    "B코스": "B코스-8.png",
    "변형B코스": "변형B코스.png",
    "C코스": "C코스-8.png",
    "D코스": "D코스-8.png",
    "E코스": "E코스-8.png",
    "F코스": "F코스-8.png",
    "관제동": "관제동.png",
    "직선코스": "직선코스.png",
    "곡선코스": "곡선코스.png",
    "위험회피": "위험회피.png",
    "소방-1": "소방01.png",
    "시가지": "시가지.png",
}

FONT_PATH = "NanumGothicBold.ttf"   # 저장소에 올린 한글 폰트 파일명


# ─────────────────────────────────────────────
# 3. 이미지/폰트 로딩 캐싱
# ─────────────────────────────────────────────
@st.cache_data
def load_base_map(path: str):
    img = Image.open(path).convert("RGB")
    return np.array(img), img.size  # (arr, (w, h))


@st.cache_data
def load_course_mask(path: str, base_size: tuple):
    img = Image.open(path).convert("RGBA")
    if img.size != base_size:
        return None, img.size

    arr = np.array(img)
    alpha = arr[:, :, 3]
    mask = (
        (alpha > 50)
        & (arr[:, :, 0] < 200)
        & (arr[:, :, 1] < 200)
        & (arr[:, :, 2] < 200)
    )
    return mask, img.size


@st.cache_data
def find_content_bbox(path: str):
    """도면 본체의 경계 상자를 찾는다.
    캔버스 테두리선 같은 '얇은 선'은 침식(erosion)으로 제거하고,
    두께가 있는 도면 본체만 남겨서 영역을 계산한다."""
    arr, (w, h) = load_base_map(path)
    nonwhite = arr.min(axis=2) < 235

    k = max(5, int(min(w, h) * 0.006))          # 이 두께보다 얇은 선은 무시
    eroded = binary_erosion(nonwhite, structure=np.ones((k, 1), dtype=bool))
    eroded = binary_erosion(eroded, structure=np.ones((1, k), dtype=bool))

    ys, xs = np.where(eroded)
    if len(xs) == 0:                             # 실패 시 원본 기준으로 대체
        ys, xs = np.where(nonwhite)
        if len(xs) == 0:
            return 0, 0, w, h
        k = 0

    pad = k // 2 + 1                             # 침식으로 줄어든 만큼 복원
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    y1 = min(h, int(ys.max()) + pad + 1)
    return x0, y0, x1, y1


@st.cache_resource
def get_font(size: int):
    candidates = [
        FONT_PATH,
        "malgunbd.ttf",
        "malgun.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


# ─────────────────────────────────────────────
# 4. 사이드바 UI
# ─────────────────────────────────────────────
st.sidebar.header("🚗 팀별 코스 배정 설정")
team_assignments = {}

for team in team_colors.keys():
    selected = st.sidebar.multiselect(
        f"{team} 주행 코스",
        options=list(course_files.keys()),
        default=[],
    )
    team_assignments[team] = selected

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 팀 색상 범례")
for team, color in team_colors.items():
    hex_color = "#%02x%02x%02x" % color
    st.sidebar.markdown(
        f"<span style='display:inline-block;width:14px;height:14px;"
        f"background-color:{hex_color};border-radius:3px;margin-right:8px;'></span>"
        f"{team}",
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")
thick_line = st.sidebar.checkbox(
    "선 두껍게 표시 (확대 시 가독성 향상)", value=True
)
auto_crop = st.sidebar.checkbox(
    "지도 여백 자동 자르기 (지도 크게 보기)", value=True
)
legend_pos = st.sidebar.selectbox("범례 위치", ["우하단", "우상단"], index=0)
legend_size = st.sidebar.slider(
    "범례 크기", min_value=0.015, max_value=0.040, value=0.024, step=0.001,
    help="값이 클수록 범례가 커집니다.",
)
img_scale = st.sidebar.slider(
    "지도 표시 크기 배율", min_value=0.5, max_value=2.0, value=1.0, step=0.1,
    help="1.0 = 원본 해상도 그대로, 2.0 = 2배 확대 표시",
)
fit_height = st.sidebar.slider(
    "지도 최대 높이 (화면 높이 %)", min_value=40, max_value=100, value=70, step=5,
    help="도면이 화면보다 길게 나오면 값을 낮추세요. 100이면 브라우저 창 높이 전체를 사용합니다.",
)

# ─────────────────────────────────────────────
# 5. 배경 지도 + 코스 선 합성 및 범례 그리기
# ─────────────────────────────────────────────
try:
    canvas_arr, (base_width, base_height) = load_base_map("blank_map.png")
    canvas_arr = canvas_arr.copy()

    active_summary = []
    overlap_tracker = np.zeros((base_height, base_width), dtype=np.uint8)
    size_error_courses = []

    for team, courses in team_assignments.items():
        team_color = team_colors[team]

        for course_name in courses:
            if course_name not in course_files:
                continue

            path = course_files[course_name]
            mask, course_size = load_course_mask(path, (base_width, base_height))

            if mask is None:
                size_error_courses.append(
                    f"{course_name} (파일 크기 {course_size} ≠ 지도 크기 {(base_width, base_height)})"
                )
                continue

            active_summary.append(f"**{team}**: {course_name}")

            if thick_line:
                mask = binary_dilation(mask, iterations=1)

            overlap_tracker[mask] += 1
            canvas_arr[mask] = team_color

    current_canvas = Image.fromarray(canvas_arr)

    # ── 1) 지도 여백 자르기 (도면 + 아주 작은 여백) ──
    if auto_crop:
        x0, y0, x1, y1 = find_content_bbox("blank_map.png")
        m = int(max(x1 - x0, y1 - y0) * 0.02)
        x0 = max(0, x0 - m)
        y0 = max(0, y0 - m)
        x1 = min(base_width, x1 + m)
        y1 = min(base_height, y1 + m)
        current_canvas = current_canvas.crop((x0, y0, x1, y1))

    # ── 2) 목표 배율로 리사이즈 (범례는 이후에 그려서 선명하게 유지) ──
    if img_scale != 1.0:
        cw, ch = current_canvas.size
        current_canvas = current_canvas.resize(
            (int(cw * img_scale), int(ch * img_scale)), Image.LANCZOS
        )

    map_w, map_h = current_canvas.size

    # ── 3) 범례 크기 계산 (지도 폭 기준 비례) ──
    SS = 2  # 슈퍼샘플링 배수
    unit = max(12, int(map_w * legend_size))
    f_title = get_font(int(unit * 1.25) * SS)
    f_item = get_font(unit * SS)

    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))

    def text_w(txt, font):
        return tmp.textlength(txt, font=font) / SS

    row_h = int(unit * 1.9)
    pad = int(unit * 0.9)
    chip = int(unit * 1.2)

    # 팀 이름 중 가장 긴 폭 → 코스 열 위치 자동 결정 (겹침 방지)
    max_name_w = max(text_w(t, f_item) for t in team_colors)
    name_x_1x = pad + chip + unit * 0.6
    course_x_1x = name_x_1x + max_name_w + unit * 0.8

    MAX_CHARS = 8
    course_w = text_w("가" * MAX_CHARS, f_item)

    box_w = int(course_x_1x + course_w + pad)
    box_h = pad * 2 + int(unit * 2.4) + row_h * len(team_colors)

    # ── 4) 캔버스 오른쪽에 범례 자리 확보 (지도를 가리지 않도록) ──
    margin = int(unit * 1.0)
    if auto_crop:
        full_w = map_w + box_w + margin * 2
        full_h = max(map_h, box_h + margin * 2)
        bg = Image.new("RGB", (full_w, full_h), (255, 255, 255))
        bg.paste(current_canvas, (0, 0))
        current_canvas = bg
    else:
        full_w, full_h = map_w, map_h

    box_x = full_w - box_w - margin
    box_y = margin if legend_pos == "우상단" else full_h - box_h - margin

    # ── 5) 범례 레이어 그리기 ──
    layer = Image.new("RGBA", (box_w * SS, box_h * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    d.rounded_rectangle(
        [0, 0, box_w * SS - 1, box_h * SS - 1],
        radius=10 * SS,
        fill=(255, 255, 255, 240),
        outline=(120, 120, 120, 255),
        width=2 * SS,
    )
    d.text((pad * SS, pad * SS), "팀별 주행 코스 현황", fill=(20, 20, 20, 255), font=f_title)

    line_y = (pad + int(unit * 1.9)) * SS
    d.line(
        [pad * SS, line_y, (box_w - pad) * SS, line_y],
        fill=(190, 190, 190, 255),
        width=2 * SS,
    )

    y = line_y + int(unit * 0.5) * SS
    name_x = int(name_x_1x * SS)
    course_x = int(course_x_1x * SS)

    for team, color in team_colors.items():
        d.rectangle(
            [
                pad * SS,
                y + int(unit * 0.15) * SS,
                (pad + chip) * SS,
                y + (int(unit * 0.15) + chip) * SS,
            ],
            fill=color + (255,),
            outline=(60, 60, 60, 255),
            width=SS,
        )
        d.text((name_x, y), team, fill=(20, 20, 20, 255), font=f_item)

        assigned = team_assignments.get(team, [])
        if assigned:
            course_str = ", ".join(assigned)
            if len(course_str) > MAX_CHARS:
                course_str = course_str[:MAX_CHARS - 1] + "…"
            txt_color = (30, 30, 30, 255)
        else:
            course_str = "미배정"
            txt_color = (150, 150, 150, 255)
        d.text((course_x, y), course_str, fill=txt_color, font=f_item)

        y += row_h * SS

    layer = layer.resize((box_w, box_h), Image.LANCZOS)

    current_canvas = current_canvas.convert("RGBA")
    current_canvas.alpha_composite(layer, (box_x, box_y))
    current_canvas = current_canvas.convert("RGB")

except FileNotFoundError as e:
    st.error(f"이미지 파일을 찾을 수 없습니다: {e}. 폴더 안에 파일들이 있는지 확인해주세요.")
    st.stop()

# ─────────────────────────────────────────────
# 6. 결과 화면 출력
# ─────────────────────────────────────────────
buf = io.BytesIO()
current_canvas.save(buf, format="PNG")
png_bytes = buf.getvalue()
b64 = base64.b64encode(png_bytes).decode()

# 도면을 먼저, 한 화면에 들어오도록 높이 제한
st.markdown(
    f"""
    <div style="text-align:center;">
        <img src="data:image/png;base64,{b64}"
             style="max-width:100%; max-height:{fit_height}vh;
                    width:auto; height:auto; object-fit:contain;
                    border:1px solid #ddd; border-radius:6px;">
    </div>
    """,
    unsafe_allow_html=True,
)

st.download_button(
    label="🖼️ 현재 현황판 이미지 다운로드 (PNG)",
    data=png_bytes,
    file_name="test_track_status.png",
    mime="image/png",
)

st.markdown("---")

# 안내 메시지는 도면 아래에 표시
if size_error_courses:
    st.error(
        "⚠️ 아래 코스 이미지가 지도와 크기가 달라 표시에서 제외되었습니다. "
        "일러스트레이터에서 동일 아트보드 기준으로 다시 내보내주세요:\n\n"
        + "\n".join(f"- {c}" for c in size_error_courses)
    )

if active_summary:
    st.info("🔹 현재 주행 중인 코스 매칭 현황:\n" + ", ".join(active_summary))
else:
    st.warning("⚠️ 사이드바에서 코스가 배정된 팀이 없습니다. 코스를 선택해주세요.")

if np.any(overlap_tracker > 1):
    st.warning("🔸 일부 구간이 2개 이상의 팀 코스와 겹칩니다. 화면에는 마지막으로 그려진 팀 색상만 표시됩니다.")