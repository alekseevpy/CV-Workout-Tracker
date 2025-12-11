import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path

# run with "streamlit run backend/app.py"

st.set_page_config(
    page_title="Workout Video Detector",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLING ====================
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f47e5;
            margin-bottom: 0.5rem;
        }
        .info-box {
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f47e5;
            margin: 1rem 0;
        }
        .error-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
        }
    </style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

def save_uploaded_file(uploaded_file):
    """Сохранить загруженный файл во временную директорию"""
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def get_video_properties(video_path):
    """Получить свойства видео"""
    cap = cv2.VideoCapture(video_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration
    }


def run_detection_on_video(video_path, model=None, progress_bar=None):
    """
    Запустить детекцию на видео
    
    Args:
        video_path: путь к видео файлу
        model: модель для детекции (если None, возвращает исходное видео)
        progress_bar: Streamlit progress bar для отслеживания прогресса
    
    Returns:
        путь к обработанному видео
    """
    cap = cv2.VideoCapture(video_path)
    
    # Получить свойства видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Создать временный файл для выходного видео
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "processed_video.mp4")
    
    # Инициализировать VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if model is not None:
            pass # todo надо сюда подгружать нашу модель
            
        
        out.write(frame)
        frame_idx += 1
        
        # Обновить progress bar
        if progress_bar is not None:
            progress = frame_idx / frame_count
            progress_bar.progress(progress)
    
    cap.release()
    out.release()
    
    return output_path


def display_video_preview(video_path, max_frames=5):
    """Показать превью видео (несколько фреймов)"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    frame_indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
    
    cols = st.columns(max_frames)
    
    for idx, col in enumerate(cols):
        frame_num = frame_indices[idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            col.image(frame_rgb)
            col.caption(f"Frame {frame_num}")
    
    cap.release()


# ==================== MAIN APP ====================

st.markdown("<h1 class='main-header'>🎥 Workout Video Detector</h1>", unsafe_allow_html=True)
st.markdown("""
    Загрузите видео с воркаут площадки, и приложение проведет детекцию тренажеров и выполняемых упражнений.
    Обработанное видео с bounding boxes будет доступно для скачивания.
""")

st.markdown("<div class='info-box'>Видео должно быть в формате MP4, MOV или AVI</div>", unsafe_allow_html=True)

# ==================== MAIN CONTENT ====================

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📁 Загрузить видео")
    uploaded_file = st.file_uploader(
        "Выберите видео файл",
        type=["mp4", "mov", "avi"],
        help="Статичное видео с воркаут площадки"
    )

with col2:
    st.subheader("📊 Информация о видео")
    video_info_placeholder = st.empty()

# ==================== VIDEO PROCESSING ====================

if uploaded_file is not None:
    st.success(f"✅ Файл загружен: {uploaded_file.name}")
    
    # Сохранить загруженный файл
    video_path = save_uploaded_file(uploaded_file)
    
    # Получить информацию о видео
    video_props = get_video_properties(video_path)
    
    with video_info_placeholder.container():
        st.markdown(f"""
            - **Разрешение:** {video_props['width']}x{video_props['height']}
            - **FPS:** {video_props['fps']}
            - **Кол-во фреймов:** {video_props['frame_count']}
            - **Длительность:** {video_props['duration']:.2f} сек
        """)
    
    # Preview видео
    st.subheader("🎬 Превью видео")
    with st.expander("Показать кадры из видео", expanded=True):
        display_video_preview(video_path, max_frames=5)
    
    st.divider()
    
    process_button = st.button(
        "🚀 Запустить детекцию",
        use_container_width=True,
        key="process_btn"
    )
    
    if process_button:
        with st.spinner("⏳ Обрабатываю видео..."):
            progress_bar = st.progress(0)
            
            output_video_path = run_detection_on_video(
                video_path,
                model=None, # todo вставить наш пайплайн
                progress_bar=progress_bar
            )
            
            progress_bar.empty()
            st.success(f"✅ Видео успешно обработано")
        
        st.subheader("📹 Обработанное видео")
        with open(output_video_path, "rb") as video_file:
            st.video(video_file)
        
        st.divider()
        
        st.subheader("💾 Скачать результат")
        with open(output_video_path, "rb") as video_file:
            st.download_button(
                label="📥 Скачать видео с детекцией",
                data=video_file.read(),
                file_name="processed_workout_video.mp4",
                mime="video/mp4",
                use_container_width=True
            )
