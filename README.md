# Uzbek NER — инференс-сервис

Сервис выделяет именованные сущности в узбекских текстах и возвращает их типы и точные символьные координаты.

Поддерживаемые классы:

- `ORG` — организации и бренды;
- `NAME` — люди;
- `GEO` — географические объекты.

Сервис реализован на FastAPI и подготовлен к запуску в Docker.

## Что нужно для запуска

На компьютере должен быть установлен Docker.

Во время сборки Docker-образа требуется доступ в интернет: в образ загружаются модель `dashakoryakovskaya/uzbek-ner` и токенизатор `FacebookAI/xlm-roberta-large`.

После сборки контейнер работает автономно: модель и все необходимые ресурсы уже находятся внутри образа, поэтому при `docker run` интернет не требуется.

## Сборка Docker-образа

Команды выполняются из корня проекта, где находится `Dockerfile`.

```bash
docker build -t ner-uz-solution .
```

Первая сборка может занять продолжительное время, так как Docker устанавливает зависимости и загружает модель внутрь образа.

После успешной сборки в конце лога должно появиться имя образа `ner-uz-solution`.

## Запуск сервиса

```bash
docker run --rm -p 8000:8000 ner-uz-solution
```

После загрузки модели сервис будет доступен по адресу:

```text
http://localhost:8000
```

Контейнер слушает `0.0.0.0:8000`.

Для остановки сервиса нажмите `Ctrl+C`.

## Проверка доступности

Обязательный endpoint:

```http
GET /healthz
```

Пример:

```bash
curl http://localhost:8000/healthz
```

Ожидаемый ответ:

```json
{"status":"ok"}
```

`/healthz` не запускает инференс и используется только для проверки готовности сервиса.

## Получение предсказаний

Обязательный endpoint:

```http
POST /api/v1/predict
```

Тело запроса — непустой JSON-массив. Каждый объект содержит уникальный `hash` и исходный `text`.

Пример запроса:

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '[
    {
      "hash": "example-001",
      "text": "Ali Toshkent shahrida ishlaydi."
    },
    {
      "hash": "example-002",
      "text": "Алишер Навоий ҳақида мақола."
    }
  ]'
```

Формат ответа:

```json
{
  "data": [
    {
      "hash": "example-001",
      "entities": [
        {"label": "NAME", "start": 0, "end": 3},
        {"label": "GEO", "start": 4, "end": 12}
      ]
    },
    {
      "hash": "example-002",
      "entities": [
        {"label": "NAME", "start": 0, "end": 13}
      ]
    }
  ]
}
```

Координаты считаются по символам Unicode и задаются полуинтервалом `[start, end)`: `start` входит в сущность, `end` указывает на первый символ после неё.

## Проверка HTTP-контракта

После запуска контейнера в отдельном терминале:

```bash
python scripts/check_service.py --url http://localhost:8000
```

Успешная проверка:

```text
OK  GET /healthz
OK  POST /api/v1/predict
Service contract: OK
```

## Проверка на validation-выборке

При необходимости можно прогнать сервис на `data/dev.jsonl` и сохранить предсказания и метрики:

```bash
python scripts/evaluate_service.py \
  --url http://localhost:8000 \
  --gold data/dev.jsonl \
  --predictions artifacts/service/dev_predictions.jsonl \
  --output artifacts/service/dev_metrics.json
```

## Модель и постобработка

В сервисе используется модель:

```text
dashakoryakovskaya/uzbek-ner
```

Токенизатор:

```text
FacebookAI/xlm-roberta-large
```

Для длинных текстов используется разбиение на перекрывающиеся окна. Предсказания на перекрытиях объединяются, после чего BIO-разметка переводится в точные символьные интервалы.

Дополнительно применяется постобработка на основе `data/train.jsonl`: уточнение границ, переопределение части меток по обучающему лексикону, обработка повторных упоминаний, удаление пересечений и дубликатов.

## Структура сервиса

Основные файлы:

```text
service/
  app.py
  predictor.py
  schemas.py

data/
  train.jsonl

Dockerfile
requirements.txt
scripts/check_service.py
scripts/evaluate_service.py
```

`streamlit_app.py` используется только как демонстрационный интерфейс и не требуется для проверки обязательного HTTP API.

## Воспроизводимый пайплайн обучения

Установка зависимостей
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Обучение
```bash
python train.py
```
