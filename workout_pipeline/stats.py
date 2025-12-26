import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os

def calculate_track_statistics(tracks, predictions):
    """
    Рассчитывает статистику по каждому треку с фильтрацией по количеству детекций
    
    Параметры:
    -----------
    tracks : dict
        Словарь с треками от функции process_video_for_lstm
    predictions : dict
        Словарь с предсказаниями от функции full_process
    
    Возвращает:
    -----------
    dict : отфильтрованная статистика по каждому треку
    DataFrame : таблица со статистикой
    """
    
    statistics = {}
    filtered_tracks = {}
    
    # Фильтрация треков по количеству детекций (сегментов)
    print(f"🔍 Фильтрация треков: минимум 3 детекции (сегмента)")
    original_count = len(tracks)
    
    for track_id in tracks.keys():
        # Количество сегментов для этого трека
        track_segments = predictions.get(track_id, [])
        segments_count = len(track_segments)
        
        if segments_count > 2:  # Больше 2 детекций
            filtered_tracks[track_id] = tracks[track_id]
            print(f"  Сохранен трек {track_id}: {segments_count} сегментов")
        else:
            print(f"  Пропущен трек {track_id}: только {segments_count} сегментов")
    
    print(f"📊 После фильтрации треков: {len(filtered_tracks)} из {original_count} треков")
    
    # Расчет статистики по отфильтрованным трекам
    for track_id, track_data in filtered_tracks.items():
        # 1. Суммарное время видимости (из tracks)
        total_time_visible = track_data.get("duration", 0)
        
        # Получаем timestamps для расчетов
        timestamps = track_data.get("timestamps", None)
        
        if timestamps is not None and len(timestamps) > 0:
            start_time = float(timestamps[0])
            end_time = float(timestamps[-1])
        else:
            start_time = 0
            end_time = 0
        
        # 2. Собираем информацию об упражнениях
        exercises_info = defaultdict(lambda: {
            "total_time": 0, 
            "segments": 0,
            "segment_details": []
        })
        
        if track_id in predictions and predictions[track_id]:
            for segment in predictions[track_id]:
                exercise = segment["predicted_class"]
                segment_duration = segment["end_time"] - segment["start_time"]
                
                # Пропускаем "no_exercise"
                if exercise == "no_exercise":
                    continue
                
                # Сохраняем детали сегмента
                segment_info = {
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "duration": segment_duration,
                    "confidence": segment.get("confidence", 0),
                    "segment_idx": segment.get("segment_idx", 0)
                }
                exercises_info[exercise]["segment_details"].append(segment_info)
        
        # 3. Фильтрация упражнений: минимум 3 детекции для учета
        filtered_exercises_info = {}
        filtered_out_exercises = []
        
        for exercise, info in exercises_info.items():
            segments_count = len(info["segment_details"])
            
            if segments_count > 2:  # Больше 2 детекций
                # Суммируем общее время упражнения
                total_time = sum(seg["duration"] for seg in info["segment_details"])
                filtered_exercises_info[exercise] = {
                    "total_time": total_time,
                    "segments": segments_count
                }
                print(f"  Трек {track_id}: упражнение '{exercise}' - {segments_count} сегментов")
            else:
                filtered_out_exercises.append(exercise)
        
        if filtered_out_exercises:
            print(f"  Трек {track_id}: пропущены упражнения {filtered_out_exercises} (≤2 сегментов)")
        
        # 4. Количество разнообразных упражнений (исключая "other")
        unique_exercises = set()
        for exercise in filtered_exercises_info.keys():
            if exercise != "other" and exercise != "no_exercise":
                unique_exercises.add(exercise)
        num_unique_exercises = len(unique_exercises)
        
        # 5. Время выполнения каждого упражнения относительно суммарного времени
        exercise_percentages = {}
        for exercise, info in filtered_exercises_info.items():
            if exercise != "other" and exercise != "no_exercise":
                percentage = (info["total_time"] / total_time_visible * 100) if total_time_visible > 0 else 0
                exercise_percentages[exercise] = {
                    "time_seconds": round(info["total_time"], 2),
                    "percentage": round(percentage, 1),
                    "segments": info["segments"]
                }
        
        # 6. Время без упражнений "other" и "no_exercise"
        total_time_without_other = total_time_visible
        if "other" in filtered_exercises_info:
            total_time_without_other -= filtered_exercises_info["other"]["total_time"]
        
        # Собираем всю статистику по треку
        statistics[track_id] = {
            "track_id": track_id,
            "total_time_visible_seconds": round(total_time_visible, 2),
            "total_time_without_other_seconds": round(total_time_without_other, 2),
            "num_unique_exercises": num_unique_exercises,
            "exercises": exercise_percentages,
            "total_segments": len(predictions.get(track_id, [])),
            "detection_percentage": track_data.get("detection_percentage", 0),
            "frames_count": len(track_data.get("keypoints_original", [])),
            "start_time": start_time,
            "end_time": end_time,
            "filtered_out_exercises": len(filtered_out_exercises),
            "filtered_out_exercises_list": filtered_out_exercises
        }
    
    # Преобразуем в DataFrame для удобства
    df_rows = []
    
    for track_id, stats in statistics.items():
        # Базовая строка
        row = {
            "track_id": track_id,
            "total_time_visible_seconds": stats["total_time_visible_seconds"],
            "total_time_without_other_seconds": stats["total_time_without_other_seconds"],
            "num_unique_exercises": stats["num_unique_exercises"],
            "total_segments": stats["total_segments"],
            "detection_percentage": stats["detection_percentage"],
            "frames_count": stats["frames_count"],
            "start_time": stats["start_time"],
            "end_time": stats["end_time"],
            "time_other_seconds": stats["total_time_visible_seconds"] - stats["total_time_without_other_seconds"],
            "other_percentage": round((stats["total_time_visible_seconds"] - stats["total_time_without_other_seconds"]) / 
                                    stats["total_time_visible_seconds"] * 100, 1) if stats["total_time_visible_seconds"] > 0 else 0,
            "filtered_out_exercises": stats["filtered_out_exercises"],
            "filtered_out_exercises_list": ", ".join(stats["filtered_out_exercises_list"]) if stats["filtered_out_exercises_list"] else "нет"
        }
        
        # Добавляем информацию по каждому упражнению
        for exercise, exercise_stats in stats["exercises"].items():
            row[f"{exercise}_time_seconds"] = exercise_stats["time_seconds"]
            row[f"{exercise}_percentage"] = exercise_stats["percentage"]
            row[f"{exercise}_segments"] = exercise_stats["segments"]
        
        df_rows.append(row)
    
    # Создаем DataFrame
    df = pd.DataFrame(df_rows)
    
    # Заполняем NaN нулями для упражнений, которые были не у всех
    df = df.fillna(0)
    
    return statistics, df

def save_statistics_to_csv(tracks, predictions, output_path="track_statistics.csv"):
    """
    Сохраняет статистику по трекам в CSV файл
    
    Параметры:
    -----------
    tracks : dict
        Словарь с треками
    predictions : dict
        Словарь с предсказаниями
    output_path : str
        Путь для сохранения CSV файла
    
    Возвращает:
    -----------
    DataFrame : таблица со статистикой
    """
    
    # Рассчитываем статистику
    statistics, df = calculate_track_statistics(tracks, predictions)
    
    # Сохраняем в CSV
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    # Выводим сводную информацию
    print(f"✅ Статистика сохранена в: {output_path}")
    print(f"📊 Всего треков: {len(statistics)}")
    
    return df, statistics

def create_statistics_report(statistics, df, output_dir="statistics_report"):
    """
    Создает и сохраняет отдельные графики с анализом статистики
    
    Параметры:
    -----------
    tracks : dict
        Словарь с треками
    predictions : dict
        Словарь с предсказаниями
    output_dir : str
        Папка для сохранения графиков
    
    Возвращает:
    -----------
    dict : пути к сохраненным файлам
    """
    
    # Создаем папку для отчетов
    os.makedirs(output_dir, exist_ok=True)
    
    # Список для хранения путей к файлам
    saved_files = {}
    
    # Настройка стиля графиков
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. График времени видимости по трекам
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='track_id', y='total_time_visible_seconds', data=df)
    ax.set_title('Время видимости по трекам', fontsize=14, fontweight='bold')
    ax.set_xlabel('Track ID', fontsize=12)
    ax.set_ylabel('Секунды', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    # Добавляем значения на столбцы
    for i, v in enumerate(df['total_time_visible_seconds']):
        ax.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    time_chart_path = os.path.join(output_dir, "01_time_by_track.png")
    plt.savefig(time_chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files["time_by_track"] = time_chart_path
    print(f"✅ Сохранен график: {time_chart_path}")
    
    # 2. График количества уникальных упражнений по трекам
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='track_id', y='num_unique_exercises', data=df)
    ax.set_title('Количество уникальных упражнений по трекам', fontsize=14, fontweight='bold')
    ax.set_xlabel('Track ID', fontsize=12)
    ax.set_ylabel('Количество упражнений', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    for i, v in enumerate(df['num_unique_exercises']):
        ax.text(i, v + 0.1, f'{v}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    exercises_chart_path = os.path.join(output_dir, "02_exercises_by_track.png")
    plt.savefig(exercises_chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files["exercises_by_track"] = exercises_chart_path
    print(f"✅ Сохранен график: {exercises_chart_path}")
    
    # 3. График процента детекции
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='track_id', y='detection_percentage', data=df)
    ax.set_title('Процент детекции по трекам', fontsize=14, fontweight='bold')
    ax.set_xlabel('Track ID', fontsize=12)
    ax.set_ylabel('Процент детекции (%)', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='Порог 80%')
    
    for i, v in enumerate(df['detection_percentage']):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.legend()
    plt.tight_layout()
    detection_chart_path = os.path.join(output_dir, "03_detection_percentage.png")
    plt.savefig(detection_chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files["detection_percentage"] = detection_chart_path
    print(f"✅ Сохранен график: {detection_chart_path}")
    
    # 4. Тепловая карта упражнений по трекам
    # Собираем данные для тепловой карты
    heatmap_data = []
    all_exercises = set()
    
    for track_id, stats in statistics.items():
        for exercise, ex_stats in stats["exercises"].items():
            heatmap_data.append({
                'track_id': track_id,
                'exercise': exercise,
                'time': ex_stats["time_seconds"]
            })
            all_exercises.add(exercise)
    
    if heatmap_data and all_exercises:
        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_pivot = heatmap_df.pivot_table(
            index='track_id', 
            columns='exercise', 
            values='time', 
            aggfunc='sum', 
            fill_value=0
        )
        
        # Сортируем упражнения по общему времени
        exercise_order = heatmap_pivot.sum().sort_values(ascending=False).index
        heatmap_pivot = heatmap_pivot[exercise_order]
        
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(
            heatmap_pivot, 
            annot=True, 
            fmt='.1f', 
            cmap='YlOrRd',
            cbar_kws={'label': 'Время (секунды)'},
            linewidths=0.5
        )
        ax.set_title('Тепловая карта: время упражнений по трекам', fontsize=14, fontweight='bold')
        ax.set_xlabel('Упражнения', fontsize=12)
        ax.set_ylabel('Track ID', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, "04_exercises_heatmap.png")
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files["exercises_heatmap"] = heatmap_path
        print(f"✅ Сохранен график: {heatmap_path}")
    
    # 5. Круговая диаграмма распределения упражнений (общая)
    if heatmap_data:
        exercise_totals = heatmap_df.groupby('exercise')['time'].sum()
        exercise_totals = exercise_totals[exercise_totals > 0]
        
        if len(exercise_totals) > 0:
            plt.figure(figsize=(10, 8))
            colors = plt.cm.Set3(np.linspace(0, 1, len(exercise_totals)))
            
            wedges, texts, autotexts = plt.pie(
                exercise_totals.values,
                labels=exercise_totals.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                pctdistance=0.85
            )
            
            plt.title('Распределение времени по упражнениям (все треки)', fontsize=14, fontweight='bold')
            
            # Улучшаем читаемость
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontsize(10)
                autotext.set_fontweight('bold')
            
            plt.tight_layout()
            pie_chart_path = os.path.join(output_dir, "05_exercises_distribution.png")
            plt.savefig(pie_chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            saved_files["exercises_distribution"] = pie_chart_path
            print(f"✅ Сохранен график: {pie_chart_path}")
    
    # 6. Stacked bar chart: время с упражнениями vs "other"
    plt.figure(figsize=(10, 6))
    
    # Подготовка данных
    track_ids = df['track_id'].astype(str)
    time_without_other = df['total_time_without_other_seconds']
    time_other = df['time_other_seconds']
    
    p1 = plt.bar(track_ids, time_without_other, label='Время с упражнениями')
    p2 = plt.bar(track_ids, time_other, bottom=time_without_other, label='Время "other"')
    
    plt.title('Соотношение времени: упражнения vs "other"', fontsize=14, fontweight='bold')
    plt.xlabel('Track ID', fontsize=12)
    plt.ylabel('Секунды', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    
    # Добавляем общее время сверху
    for i, (total, wo) in enumerate(zip(df['total_time_visible_seconds'], time_without_other)):
        plt.text(i, total + 0.5, f'{total:.1f}', ha='center', va='bottom', fontsize=9)
        # Процент времени с упражнениями
        percentage = (wo / total * 100) if total > 0 else 0
        plt.text(i, wo/2, f'{percentage:.0f}%', ha='center', va='center', 
                color='white', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    stacked_chart_path = os.path.join(output_dir, "06_time_distribution.png")
    plt.savefig(stacked_chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files["time_distribution"] = stacked_chart_path
    print(f"✅ Сохранен график: {stacked_chart_path}")
    
    # 7. График количества сегментов по трекам
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='track_id', y='total_segments', data=df)
    ax.set_title('Количество сегментов по трекам', fontsize=14, fontweight='bold')
    ax.set_xlabel('Track ID', fontsize=12)
    ax.set_ylabel('Количество сегментов', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    for i, v in enumerate(df['total_segments']):
        ax.text(i, v + 0.1, f'{v}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    segments_chart_path = os.path.join(output_dir, "07_segments_by_track.png")
    plt.savefig(segments_chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files["segments_by_track"] = segments_chart_path
    print(f"✅ Сохранен график: {segments_chart_path}")
    
    # 8. Box plot времени видимости
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df['total_time_visible_seconds'])
    plt.title('Распределение времени видимости треков', fontsize=14, fontweight='bold')
    plt.ylabel('Секунды', fontsize=12)
    
    # Добавляем среднее значение
    mean_time = df['total_time_visible_seconds'].mean()
    plt.axhline(y=mean_time, color='r', linestyle='--', alpha=0.7, 
                label=f'Среднее: {mean_time:.1f} сек')
    
    # Добавляем точки для каждого трека
    for i, time in enumerate(df['total_time_visible_seconds']):
        plt.scatter(0, time, alpha=0.6, s=50)
    
    plt.legend()
    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, "08_time_distribution_boxplot.png")
    plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files["time_boxplot"] = boxplot_path
    print(f"✅ Сохранен график: {boxplot_path}")
    
    return saved_files, df, statistics