# Работа с ветками

## 1. Клонируем репозиторий и переходим на свою ветку:

    ```bash
    $ git clone <HTTPS/SSH>
    $ cd CV-Workout-Tracker
    $ git checkout <имя_вашей_ветки>
    ```

## 2. Синхронизируем локальный репозиторий с удалённым:

    ```bash
    $ git fetch origin
    $ git pull origin main
    $ git pull origin <имя_вашей_ветки>
    ```

## 3. Фиксируем изменения, коммитим и делаем push:

    ```bash
    $ git add .
    $ git commit -m "Описание изменений"
    $ git push origin <имя_вашей_ветки>
    ```