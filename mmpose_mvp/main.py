from pathlib import Path

from mmpose_mvp.run_pushups_pipeline import run_pushups_pipeline

BASE_DIR = Path("mmpose_mvp/data_mvp")

GEMINI_API_KEY = "AIzaSyB2l1Q8zC38PqeY_uQP7g9KNP9msxgQDsk"

result = run_pushups_pipeline(
    user_video_src="notebooks/mmpose/input_compare_angles/5 отжиманий.mp4",  # ОБЯЗАТЕЛЬНО, сейчас тут пример с локалки
    reference_video_src="notebooks/mmpose/input_compare_angles/RARE_PUSH-UPS_INPUT.mp4",  # можно None, если уже есть эталон, сейчас тут пример с локалки
    base_dir=BASE_DIR,
    gemini_api_key=GEMINI_API_KEY,
    device="cpu",
)

print("DONE")
print("Comparison plot:", result.comparison_png)
print("Coach answer:", result.coach_answer_txt)

# График отдать пользователю, будет на пути /mmpose_mvp/data_mvp/pushups/user/comparison_profiles.png
# Рекомендации от Gemini будут на пути /mmpose_mvp/data_mvp/pushups/user/_coach/coach_answer.txt