import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path

# run with "streamlit run backend/app.py"

st.set_page_config(
    page_title="Активность в воркаут зоне",
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
    # output_path = os.path.join(temp_dir, "processed_video.mp4")
    output_path = os.path.join(temp_dir, "processed_video.webm")
    
    # Инициализировать VideoWriter
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
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

def run_pose_estimation_on_video(video_path, model=None, progress_bar=None):
    """
    Запустить анализ техники / позы на видео.

    Args:
        video_path: путь к видео файлу
        model: модель позовой оценки (возвращает видео со скелетом и текст)
        progress_bar: Streamlit progress bar для отслеживания прогресса

    Returns:
        (путь к обработанному видео, текст с рекомендацией)
    """
    cap = cv2.VideoCapture(video_path)

    # Получить свойства видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Создать временный файл для выходного видео
    temp_dir = tempfile.gettempdir()
    # output_path = os.path.join(temp_dir, "pose_video.mp4")
    output_path = os.path.join(temp_dir, "pose_video.webm")
    
    # Инициализировать VideoWriter
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0

    # Заглушка для текстовых рекомендаций
    recommendations = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if model is not None:
            pass # todo надо сюда подгружать нашу модель

        out.write(frame)
        frame_idx += 1

        # Обновить progress bar
        if progress_bar is not None and frame_count > 0:
            progress = frame_idx / frame_count
            progress_bar.progress(progress)

    cap.release()
    out.release()

    # Если модели нет, возвращаем дефолтный текст
    if not recommendations:
        recommendations_text = (
            "Модель анализа техники пока не подключена."
        )
    else:
        # Уникализируем и склеиваем комментарии
        recommendations = list(dict.fromkeys(recommendations))
        recommendations_text = "\n".join(f"- {r}" for r in recommendations)

    return output_path, recommendations_text


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

st.markdown("<h1 class='main-header'>Проект для отслеживания активности в воркаут зоне<br/>и оценки качества выполнения упражнения</h1>", unsafe_allow_html=True)
st.markdown("""
    - **Вкладка "Детекция объектов":** Загрузите видео с воркаут площадки, и приложение проведет детекцию тренажеров и выполняемых упражнений. Обработанное видео с bounding boxes будет доступно для скачивания.
    - **Вкладка "Анализ техники":** Загрузите видео с выполнением упражнения, и приложение проведет анализ техники выполнения упражнений (pose estimation + рекомендации)
""")

st.markdown("<div class='info-box'>Видео должно быть в формате MP4, MOV или AVI</div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Детекция объектов", "Анализ техники"])

# ==================== TAB 1: DETECTION ====================
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Загрузить видео")
        uploaded_file = st.file_uploader(
            "Выберите видео файл",
            type=["mp4", "mov", "avi"],
            help="Статичное видео с воркаут площадки"
        )

    with col2:
        st.subheader("Информация о видео")
        video_info_placeholder = st.empty()

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
        
        st.subheader("Превью видео")
        display_video_preview(video_path, max_frames=5)
        
        st.divider()
        
        process_button = st.button(
            "Запустить детекцию",
            use_container_width=True,
            key="process_btn"
        )
        
        if "detection_processed_path" not in st.session_state:
            st.session_state.detection_processed_path = None

        if process_button:
            with st.spinner("Обрабатываю видео..."):
                progress_bar = st.progress(0)
                
                output_video_path = run_detection_on_video(
                    video_path,
                    model=None, # todo вставить наш пайплайн
                    progress_bar=progress_bar
                )
                
                progress_bar.empty()
                st.success(f"✅ Видео успешно обработано")
            st.session_state.detection_processed_path = output_video_path

        if st.session_state.detection_processed_path is not None:
            st.subheader("Обработанное видео")
            with open(st.session_state.detection_processed_path, "rb") as video_file:
                st.video(video_file)
            
            st.divider()
            
            with open(st.session_state.detection_processed_path, "rb") as video_file:
                st.download_button(
                    label="Скачать видео",
                    data=video_file.read(),
                    file_name="processed_workout_video.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )

# ==================== TAB 2: POSE ANALYSIS ====================
with tab2:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Загрузить видео упражнения")
        uploaded_pose_file = st.file_uploader(
            "Выберите видео файл с выполнением упражнения",
            type=["mp4", "mov", "avi"],
            help="Видео одного человека, выполняющего упражнение",
            key="pose_video_uploader"
        )

    with col2:
        st.subheader("Информация о видео")
        pose_video_info_placeholder = st.empty()

    if uploaded_pose_file is not None:
        st.success(f"✅ Файл загружен: {uploaded_pose_file.name}")

        pose_video_path = save_uploaded_file(uploaded_pose_file)
        pose_video_props = get_video_properties(pose_video_path)

        with pose_video_info_placeholder.container():
            st.markdown(f"""
                - **Разрешение:** {pose_video_props['width']}x{pose_video_props['height']}
                - **FPS:** {pose_video_props['fps']}
                - **Кол-во фреймов:** {pose_video_props['frame_count']}
                - **Длительность:** {pose_video_props['duration']:.2f} сек
            """)

        st.subheader("Превью видео")
        display_video_preview(pose_video_path, max_frames=5)

        st.divider()

        analyze_button = st.button(
            "Запустить анализ техники",
            use_container_width=True,
            key="analyze_pose_btn"
        )

        if "pose_processed_path" not in st.session_state:
            st.session_state.pose_processed_path = None

        if analyze_button:
            with st.spinner("Анализирую технику..."):
                progress_bar = st.progress(0)
                
                pose_video_out_path, recommendations_text = run_pose_estimation_on_video(
                    pose_video_path,
                    model=None, # todo нашу модель вставить
                    progress_bar=progress_bar
                )

                progress_bar.empty()
                st.success(f"✅ Видео успешно обработано")
            st.session_state.pose_processed_path = pose_video_out_path
            st.session_state.pose_recommendations_text = recommendations_text

        if st.session_state.pose_processed_path is not None:                
            col_video, col_text = st.columns([2, 1])

            with col_video:
                st.subheader("Видео с определением позы")
                with open(st.session_state.pose_processed_path, "rb") as video_file:
                    st.video(video_file)

            with col_text:
                st.subheader("Рекомендации по выполнению")
                st.text(st.session_state.pose_recommendations_text)

            st.divider()

            with open(st.session_state.pose_processed_path, "rb") as video_file:
                st.download_button(
                    label="Скачать видео",
                    data=video_file.read(),
                    file_name="pose_workout_video.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )