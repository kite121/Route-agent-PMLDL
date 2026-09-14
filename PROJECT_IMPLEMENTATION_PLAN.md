# AgentRoute: план реализации проекта

Этот документ фиксирует согласованный порядок реализации проекта **Route-agent-PMLDL**. Его следует использовать как основной чек-лист: перед началом следующего этапа необходимо проверить критерии готовности предыдущего.

## Общие правила работы

- [ ] Не переходить к следующему этапу, пока текущий этап не проверен.
- [ ] После каждого логически завершённого этапа делать отдельный Git commit.
- [ ] Не добавлять в Git датасеты, обученные модели, секреты, логи Airflow и другие генерируемые артефакты.
- [ ] Загружать данные воспроизводимо через `download_data.py`, а не вручную.
- [ ] Хранить секреты только в локальном `.env`; в Git добавлять только `.env.example`.
- [ ] Использовать `requirements.txt`; `pyproject.toml` не использовать.
- [ ] Не добавлять `LICENSE`, `notebooks/`, `tests/`, `confusion_matrix.png` и LiteLLM.
- [ ] Streamlit должен обращаться к FastAPI и не должен загружать модель самостоятельно.
- [ ] Все эксперименты, включая отклонённые quality gate, сохранять в ClearML.

## Текущий прогресс

| Этап | Статус |
|---|---|
| Структура репозитория | Готова |
| 0. Базовая настройка | Готова |
| 1. Data Engineering | Не начат |
| 2. Model Engineering | Не начат |
| 3. FastAPI и Streamlit | Не начат |
| 4. Docker deployment | Не начат |
| 5. Airflow | Не начат |
| 6. Финализация окружения и команд | Не начат |
| 7. README | Не начат |

## Этап 0. Базовая настройка проекта

### Задачи

- [x] Заполнить `.gitignore` до загрузки данных и создания локального `.env`.
- [x] Заполнить `.env.example` названиями необходимых переменных без секретных значений.
- [x] Создать локальный `.env`, который не попадёт в Git.
- [x] Создать локальное `.venv` на Python 3.12.
- [x] Создать предварительный корневой `requirements.txt`.
- [x] Настроить `configs/training.yaml`.
- [x] Настроить `configs/tools.yaml`.
- [x] Настроить `configs/logging.yaml`.

На этом этапе `requirements.txt` остаётся предварительным. Окончательные версии зависимостей фиксируются после реализации и проверки всех компонентов.

### Критерии готовности

- [x] Секреты и генерируемые файлы исключены из Git.
- [x] Все YAML-файлы корректно читаются.
- [x] Реестр инструментов и основные параметры обучения определены.
- [x] Новое окружение можно создать через предварительный `requirements.txt`.

## Этап 1. Data Engineering и батчи

### Порядок реализации

```text
code/datasets/download_data.py
        ↓
code/datasets/create_batches.py
        ↓
code/datasets/validate_data.py
        ↓
code/datasets/prepare_data.py
```

`code/datasets/schemas.py` содержит единые схемы и контракты данных для всех операций этапа.

### Задачи

- [ ] Зафиксировать источник и точную версию Amazon MASSIVE 1.1.
- [ ] Реализовать воспроизводимую загрузку данных в `download_data.py`.
- [ ] Сформировать фиксированный test-набор.
- [ ] Разделить training pool на последовательные батчи.
- [ ] Присвоить каждому батчу уникальный `batch_id`.
- [ ] Рассчитать SHA256 для каждого батча.
- [ ] Реализовать проверку обязательных полей и типов.
- [ ] Удалить пропуски, дубликаты и некорректные примеры.
- [ ] Обработать аномально короткие и длинные запросы.
- [ ] Проверить соответствие меток реестру `configs/tools.yaml`.
- [ ] Сформировать train, validation и test.
- [ ] Создать отчёт обработки данных.
- [ ] Создать версию датасета и manifest.
- [ ] При необходимости зарегистрировать версию через ClearML Data.

### Проверки

- [ ] Загружается правильная версия MASSIVE.
- [ ] Батчи создаются в ожидаемом порядке.
- [ ] Повторный запуск не создаёт дубликаты батчей.
- [ ] Один и тот же батч не обрабатывается дважды.
- [ ] Фиксированный test не попадает в train.
- [ ] Train, validation и test не пересекаются.
- [ ] Количество удалённых строк отражено в отчёте.
- [ ] Hashes и количество строк отражены в manifest.
- [ ] При отсутствии нового батча pipeline корректно определяет `no_new_data`.

### Критерии готовности

- [ ] Все data-скрипты успешно выполняются последовательно.
- [ ] Data pipeline воспроизводимо создаёт обработанные данные из зафиксированного источника.
- [ ] Результат каждого запуска связан с конкретной версией и hash датасета.

## Этап 2. Model Engineering и ClearML

### Порядок реализации

```text
code/models/build_training_pairs.py
        ↓
code/models/train.py
        ↓
code/models/evaluate.py
        ↓
code/models/quality_gate.py
        ↓
code/models/package_model.py
        ↓
code/models/inference.py
```

### Задачи

- [ ] Сформировать пары `query → tool description`.
- [ ] Сформировать negative examples.
- [ ] Добавить необходимые E5-префиксы.
- [ ] Реализовать загрузку `intfloat/multilingual-e5-small`.
- [ ] Реализовать fine-tuning эмбеддера.
- [ ] Начинать scheduled training с зафиксированного base checkpoint.
- [ ] Настроить ClearML Task для каждого запуска.
- [ ] Логировать конфигурацию обучения, seed и Git commit.
- [ ] Логировать dataset version, hashes и Airflow run ID.
- [ ] Логировать training и validation metrics.
- [ ] Рассчитать `Accuracy@1`.
- [ ] Рассчитать `Recall@3`.
- [ ] Рассчитать `MRR`.
- [ ] Рассчитать `Macro F1`.
- [ ] Проверить качество fallback/no-tool.
- [ ] Измерить время inference.
- [ ] Сравнить base model с fine-tuned model.
- [ ] Реализовать quality gate.
- [ ] Не деплоить модель, которая не прошла quality gate.
- [ ] Сохранять отклонённую модель и эксперимент в ClearML с соответствующим статусом или тегом.
- [ ] Упаковать принятую модель в `agent_router.tar.gz`.
- [ ] Создать `model_manifest.json` с lineage модели.
- [ ] Реализовать локальный inference из упакованного артефакта.

### Проверки

- [ ] Обучающие пары построены корректно.
- [ ] Fine-tuning выполняется без ошибок.
- [ ] Все обязательные параметры и метрики видны в ClearML.
- [ ] Model artifact загружен в ClearML.
- [ ] Отклонённая модель не заменяет текущую champion model.
- [ ] Принятая модель становится кандидатом на deployment.
- [ ] `model_manifest.json` содержит model version, base model, dataset version, hashes, Git commit, ClearML Task ID и метрики.
- [ ] Упакованная модель загружается и выполняет prediction локально.

### Критерии готовности

- [ ] Полный model pipeline работает без API, Streamlit и Airflow.
- [ ] Эксперимент полностью воспроизводим по ClearML Task и manifests.
- [ ] Quality gate однозначно определяет, можно ли разворачивать модель.

## Этап 3. FastAPI и Streamlit

### 3.1 FastAPI

Реализовать:

```text
code/deployment/api/main.py
code/deployment/api/model_service.py
code/deployment/api/schemas.py
```

### Задачи и проверки API

- [ ] Реализовать загрузку champion model.
- [ ] Реализовать `GET /health`.
- [ ] Реализовать `GET /model-info`.
- [ ] Реализовать `POST /predict`.
- [ ] Возвращать выбранный tool, score, top-k, решение route/fallback и model version.
- [ ] Валидировать входные данные.
- [ ] Проверить корректные запросы.
- [ ] Проверить пустые и некорректные запросы.
- [ ] Проверить загрузку именно той версии модели, которая указана в `model_manifest.json`.

### 3.2 Streamlit

Реализовать:

```text
code/deployment/app/app.py
code/deployment/app/api_client.py
```

### Задачи и проверки приложения

- [ ] Добавить поле пользовательского запроса.
- [ ] Добавить выбор `top_k`.
- [ ] Добавить кнопку запуска prediction.
- [ ] Показать выбранный tool и similarity score.
- [ ] Показать альтернативные tools.
- [ ] Показать route/fallback и model version.
- [ ] Показать понятную ошибку при недоступности API.
- [ ] Убедиться, что Streamlit обращается к FastAPI.
- [ ] Убедиться, что Streamlit не импортирует и не загружает модель.

### Критерии готовности

- [ ] API и приложение работают локально как отдельные процессы.
- [ ] Prediction проходит по цепочке `Streamlit → FastAPI → model`.

## Этап 4. Docker deployment

### Задачи

- [ ] Заполнить `code/deployment/api/requirements.txt`.
- [ ] Заполнить `code/deployment/app/requirements.txt`.
- [ ] Подготовить `code/deployment/api/Dockerfile`.
- [ ] Подготовить `code/deployment/app/Dockerfile`.
- [ ] Подготовить `code/deployment/docker-compose.yml`.
- [ ] Настроить внутренний адрес API как `http://api:8000`.
- [ ] Добавить Docker healthcheck для API.
- [ ] Реализовать `code/deployment/smoke_check.py`.
- [ ] Убедиться, что API image получает конкретную champion model.

### Проверки

- [ ] API image успешно собирается.
- [ ] Streamlit image успешно собирается.
- [ ] API и приложение запускаются в отдельных контейнерах.
- [ ] Контейнер приложения видит API по имени сервиса `api`.
- [ ] `GET /health` возвращает успешный статус.
- [ ] `GET /model-info` возвращает ожидаемую версию модели.
- [ ] `POST /predict` работает внутри Docker deployment.
- [ ] Streamlit показывает prediction, полученный от API.
- [ ] Smoke check завершается успешно.

### Критерии готовности

- [ ] Полный deployment работает командой Docker Compose без участия Airflow.
- [ ] API и приложение удовлетворяют обязательному требованию отдельных Docker-контейнеров.

## Этап 5. Airflow и автоматизация

Airflow подключается только после того, как все команды data, model и deployment работают отдельно.

### Итоговая последовательность DAG

```text
download data / find next batch
        ↓
validate and prepare data
        ↓
build training pairs
        ↓
train and evaluate
        ↓
quality gate
        ↓
package accepted model
        ↓
build and start Docker services
        ↓
deployment smoke check
```

### Задачи

- [ ] Реализовать `services/airflow/dags/agent_router_pipeline.py`.
- [ ] Разделить DAG на Data Engineering, Model Engineering и Deployment.
- [ ] Передавать между задачами только небольшие metadata, ID и пути к файлам.
- [ ] Не передавать датасеты и model artifacts через XCom.
- [ ] Настроить `schedule="*/5 * * * *"` или больший обоснованный интервал.
- [ ] Настроить `catchup=False`.
- [ ] Настроить `max_active_runs=1`.
- [ ] Настроить retry и понятное логирование ошибок.
- [ ] Пропускать обучение при отсутствии новых данных в production-режиме.
- [ ] Для демонстрации подготовить достаточное количество батчей для полного scheduled run.

### Проверки

- [ ] DAG импортируется без ошибок.
- [ ] DAG успешно запускается вручную.
- [ ] Каждая задача отображается в Airflow UI.
- [ ] Зависимости задач соответствуют трём стадиям assignment.
- [ ] Scheduled run запускается автоматически.
- [ ] Два запуска обучения не выполняются одновременно.
- [ ] Ошибка quality gate не заменяет champion model.
- [ ] Deployment использует артефакт текущего успешного запуска.
- [ ] Финальный smoke check подтверждает работу новой версии.

### Критерии готовности

- [ ] Полный pipeline автоматически проходит все три обязательные стадии.
- [ ] Результат scheduled run виден в Airflow, ClearML, FastAPI и Streamlit.

## Этап 6. Финализация окружения и команд

### Задачи

- [ ] Удалить неиспользуемые зависимости.
- [ ] Зафиксировать окончательные версии в корневом `requirements.txt`.
- [ ] Проверить service-specific requirements API и Streamlit.
- [ ] Проверить `.env.example` и удалить любые реальные секреты.
- [ ] Повторно проверить `.gitignore`.
- [ ] Заполнить `Makefile` только проверенными командами.

### Планируемые Make-команды

```text
make install
make data
make train
make api
make app
make compose-up
make airflow-test
make smoke
```

### Критерии готовности

- [ ] Проект устанавливается в чистом окружении.
- [ ] Все Make-команды выполняют документированные операции.
- [ ] В Git отсутствуют секреты, датасеты, модели, кэши и runtime-логи.

## Этап 7. README и финальная демонстрация

### README должен содержать

- [ ] Назначение проекта.
- [ ] Краткое описание agent tool routing.
- [ ] Архитектуру трёх стадий pipeline.
- [ ] Описание Amazon MASSIVE 1.1 и batching.
- [ ] Инструкцию настройки ClearML.
- [ ] Описание переменных окружения.
- [ ] Инструкцию создания окружения.
- [ ] Команды запуска Data Engineering.
- [ ] Команды запуска Model Engineering.
- [ ] Команды локального запуска API и Streamlit.
- [ ] Команду Docker Compose.
- [ ] Инструкцию запуска Airflow.
- [ ] Расписание pipeline.
- [ ] Адреса API, Streamlit, Airflow и ClearML.
- [ ] Описание model quality gate.
- [ ] Описание воспроизводимости через Git commit, dataset version и hashes.
- [ ] Сценарий демонстрации assignment.

### Финальная проверка assignment

- [ ] Data Engineering реализован и работает.
- [ ] Model Engineering реализован и работает.
- [ ] Метрики и модели логируются в ClearML.
- [ ] API и приложение работают в отдельных Docker-контейнерах.
- [ ] Streamlit показывает prediction, полученный от FastAPI.
- [ ] Полный pipeline запускается автоматически по расписанию.
- [ ] Репозиторий имеет логичную структуру.
- [ ] README позволяет запустить проект с чистого окружения.

## Правило перехода между этапами

Следующий этап начинается только после выполнения всех критериев готовности текущего этапа. Если во время реализации меняется архитектурное решение, изменение сначала фиксируется в этом документе, а затем применяется в коде.
