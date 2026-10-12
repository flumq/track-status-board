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

# 2. 팀 리스트 및 팀별 지정 색상 (RGB)
team_colors = {
    "Team 1": (0, 102, 204),    # 파란색
    "Team 2": (220, 50, 50),    # 빨간색
    "Team 3": (240, 180, 0),    # 노란색
    "Team 4": (46, 139, 87),    # 초록색
    "Team 5": (138, 43, 226),   # 보라색
    "공사 및 작업": (255, 110, 0)     # 비비드 오렌지
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
    "시가지": "시가지.png"
}

# ─────────────────────────────────────────────
# 3. 이미지 로딩 캐싱
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

# 고해상도 출력을 위한 폰트 로드 함수
@st.cache_resource
def get_font(size: int):
    import platform
    system = platform.system()
    try:
        if system == "Windows":
            return ImageFont.truetype("malgun.ttf", size)
        elif system == "Darwin": # Mac
            return ImageFont.truetype("AppleGothic", size)
        else: # Linux / Streamlit Cloud 등
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except IOError:
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

    current_canvas = Image.fromarray(canvas_arr)

    # ─────────────────────────────────────────────
    # 지도 이미지 '내부' 우측 상단 빈 공간에 선명한 오버레이 범례 그리기
    # ─────────────────────────────────────────────
    draw = ImageDraw.Draw(current_canvas)
    
    font_title = get_font(32)
    font_item = get_font(26)

    box_w = 420
    box_h = 90 + (len(team_colors) * 52)
    box_x = base_width - box_w - 40  
    box_y = 40                       

    # 배경 박스 및 테두리
    draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=(255, 255, 255), outline=(150, 150, 150), width=3)

    # 범례 제목
    draw.text((box_x + 24, box_y + 20), "🏁 팀별 주행 코스 현황", fill=(20, 20, 20), font=font_title)
    
    # 구분선
    draw.line([box_x + 20, box_y + 70, box_x + box_w - 20, box_y + 70], fill=(200, 200, 200), width=2)

    # 각 팀별 색상칩 및 코스 표시
    current_y = box_y + 85
    for team, color in team_colors.items():
        draw.rectangle([box_x + 24, current_y + 6, box_x + 54, current_y + 36], fill=color, outline=(80, 80, 80), width=1)
        draw.text((box_x + 70, current_y), team, fill=(20, 20, 20), font=font_item)
        
        assigned_courses = team_assignments.get(team, [])
        if assigned_courses:
            course_str = ", ".join(assigned_courses)
            if len(course_str) > 15:
                course_str = course_str[:14] + "..."
        else:
            course_str = "미배정"

        draw.text((box_x + 180, current_y), f": {course_str}", fill=(80, 80, 80), font=font_item)
        
        current_y += 52

    if img_scale != 1.0:
        new_size = (int(base_width * img_scale), int(base_height * img_scale))
        current_canvas = current_canvas.resize(new_size, Image.LANCZOS)

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