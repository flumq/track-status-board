import io

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation

# ─────────────────────────────────────────────
# 1. 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="테스트 트랙 코스 현황판",
    page_icon="🏁",
    layout="wide",
)

st.title("🏁 테스트 트랙 팀-코스 매칭 실시간 현황판")
st.write("사이드바에서 각 팀이 주행할 코스를 선택하시면, 지도 내부 오버레이 범례에 실시간으로 반영됩니다.")

# ─────────────────────────────────────────────
# 2. 팀 리스트 및 팀별 지정 색상 (RGB)
# ─────────────────────────────────────────────
team_colors = {
    "Team 1": (0, 102, 204),        # 파란색
    "Team 2": (220, 50, 50),        # 빨간색
    "Team 3": (240, 180, 0),        # 노란색
    "Team 4": (46, 139, 87),        # 초록색
    "Team 5": (138, 43, 226),       # 보라색
    "공사 및 작업": (255, 110, 0),   # 비비드 오렌지
}

# 코스별 이미지 파일명 매핑
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

# 저장소에 함께 올린 한글 폰트 파일명
FONT_PATH = "NanumGothicBold.ttf"


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


@st.cache_resource
def get_font(size: int):
    """한글 지원 폰트를 우선순위대로 탐색"""
    candidates = [
        FONT_PATH,                                                  # 저장소에 포함한 폰트
        "malgunbd.ttf",                                             # Windows
        "malgun.ttf",                                               # Windows
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",               # Mac
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",      # Linux (nanum 설치 시)
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
img_scale = st.sidebar.slider(
    "지도 표시 크기 배율", min_value=0.5, max_value=2.0, value=1.0, step=0.1,
    help="1.0 = 원본 해상도 그대로, 2.0 = 2배 확대 표시",
)

# ─────────────────────────────────────────────
# 5. 배경 지도 + 코스 선 합성 및 오버레이 범례 그리기
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

    # ── 1) 먼저 지도를 목표 배율로 리사이즈 ──
    current_canvas = Image.fromarray(canvas_arr)
    if img_scale != 1.0:
        new_size = (int(base_width * img_scale), int(base_height * img_scale))
        current_canvas = current_canvas.resize(new_size, Image.LANCZOS)

    W, H = current_canvas.size

    # ── 2) 리사이즈된 지도 위에 범례를 고해상도 RGBA 레이어로 그림 ──
    unit = max(14, int(W * 0.018))          # 지도 폭 기준 기본 글자 단위
    row_h = int(unit * 1.9)
    pad = int(unit * 0.9)
    chip = int(unit * 1.2)
    box_w = int(unit * 17)
    box_h = pad * 2 + int(unit * 2.4) + row_h * len(team_colors)
    box_x = W - box_w - int(unit * 1.5)
    box_y = int(unit * 1.5)

    # 2배 슈퍼샘플링으로 그린 뒤 축소 → 안티앨리어싱으로 선명하게
    SS = 2
    layer = Image.new("RGBA", (box_w * SS, box_h * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f_title = get_font(int(unit * 1.25) * SS)
    f_item = get_font(unit * SS)

    d.rounded_rectangle(
        [0, 0, box_w * SS - 1, box_h * SS - 1],
        radius=12 * SS,
        fill=(255, 255, 255, 235),
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
    name_x = (pad + chip + int(unit * 0.6)) * SS
    course_x = (pad + chip + int(unit * 5.2)) * SS

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
            if len(course_str) > 9:
                course_str = course_str[:8] + "…"
            txt_color = (30, 30, 30, 255)
        else:
            course_str = "미배정"
            txt_color = (150, 150, 150, 255)
        d.text((course_x, y), course_str, fill=txt_color, font=f_item)

        y += row_h * SS

    layer = layer.resize((box_w, box_h), Image.LANCZOS)

    # RGBA 합성 후 다시 RGB로
    current_canvas = current_canvas.convert("RGBA")
    current_canvas.alpha_composite(layer, (box_x, box_y))
    current_canvas = current_canvas.convert("RGB")

except FileNotFoundError as e:
    st.error(f"이미지 파일을 찾을 수 없습니다: {e}. 폴더 안에 파일들이 있는지 확인해주세요.")
    st.stop()

# ─────────────────────────────────────────────
# 6. 결과 화면 출력
# ─────────────────────────────────────────────
st.markdown("---")

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

st.image(current_canvas, caption="테스트 트랙 종합 오버레이 현황판", use_container_width=True)

buf = io.BytesIO()
current_canvas.save(buf, format="PNG")
st.download_button(
    label="🖼️ 현재 현황판 이미지 다운로드 (PNG)",
    data=buf.getvalue(),
    file_name="test_track_status.png",
    mime="image/png",
)