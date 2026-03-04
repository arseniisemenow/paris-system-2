# Topic Matcher

Инструмент для сопоставления тематик из различных источников: академические публикации, профессиональные сообщества и массовая информация.

## Что умеет проект

- 📥 **Сбор данных** из различных источников
  - arXiv API (академические публикации)
  - Habr RSS (профессиональные статьи)
  - Легко добавить новые источники

- 🔍 **Тематическое моделирование**
  - LDA (Latent Dirichlet Allocation)
  - Извлечение ключевых слов из текста

- 🔄 **Сопоставление тем**
  - Сравнение тем между источниками
  - Jaccard similarity и косинусное сходство
  - Выявление общих тем и уникальных

- 📊 **Веб-интерфейс**
  - Дашборд со статистикой
  - Просмотр тем и статей
  - Тепловая карта схожести тем

## Скриншоты

### Обзор
<!-- Add screenshot here: docs/images/overview.png -->
![Overview-placeholder]

### Темы
<!-- Add screenshot here: docs/images/topics.png -->
![Topics-placeholder]

### Сопоставление
<!-- Add screenshot here: docs/images/comparison.png -->
![Comparison-placeholder]

### Статьи
<!-- Add screenshot here: docs/images/articles.png -->
![Articles-placeholder]

## Быстрый старт

```bash
# Установка зависимостей
pip install -r requirements.txt

# Активация виртуального окружения
source venv/bin/activate

# Запуск полного анализа
python main.py

# Запуск веб-интерфейса
streamlit run ui/app.py
```

## Структура проекта

```
topic-matcher/
├── config.py                 # Конфигурация
├── database.py               # SQLite база данных
├── models.py                 # Модели данных
├── main.py                   # Точка входа
│
├── collectors/               # Сборщики данных
│   ├── arxiv_collector.py
│   └── habr_collector.py
│
├── analyzer/                 # Анализ
│   ├── preprocessing.py
│   ├── topic_model.py
│   └── comparison.py
│
├── ui/                      # Интерфейс
│   └── app.py              # Streamlit
│
└── tests/                   # Тесты
```

## Команды

| Команда | Описание |
|---------|----------|
| `python main.py` | Полный анализ |
| `python main.py --collect-only` | Только сбор данных |
| `python main.py --analyze-only` | Только анализ тем |
| `python main.py --compare-only` | Только сравнение |
| `streamlit run ui/app.py` | Запуск UI |

## Тесты

```bash
pytest tests/ -v
```

## Технологии

- Python 3.12+
- SQLite
- scikit-learn (LDA)
- Streamlit
- Plotly

## Лицензия

MIT
