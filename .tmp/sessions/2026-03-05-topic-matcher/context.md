# Task Context: Topic Matcher MVP

Session ID: 2026-03-05-topic-matcher
Created: 2026-03-05T00:00:00Z
Status: in_progress

## Current Request
Создать агентную систему на Python для сопоставления тематик популярных в профессиональных сообществах, массовой информации и академических публикаций. MVP с минимальным функционалом.

## Context Files (Standards to Follow)
- `/home/arseni/.config/opencode/context/core/standards/code-quality.md` — чистый код, pure functions, иммутабельность, модульность
- `/home/arseni/.config/opencode/context/core/standards/test-coverage.md` — AAA паттерн, тестирование, покрытие

## Reference Files (Source Material)
- Проект на Python без существующей кодовой базы
- Задачи: парсинг, анализ текста, тематическое моделирование, визуализация

## External Docs Needed
- requests (для HTTP)
- scikit-learn (для LDA)
- streamlit (для UI)
- pymorphy3 / natasha (для русской морфологии)

## Components
1. **Infrastructure** — config, database (SQLite), models
2. **Collectors** — arXiv API, парсинг Хабра
3. **Analyzer** — preprocessing, topic modeling (LDA), comparison
4. **UI** — Streamlit дашборд

## Constraints
- MVP: простое, рабочее, без продвинутых фич
- База данных: SQLite
- Парсинг: официальные API + бережный HTML парсинг
- Тематическое моделирование: LDA (проще чем BERTopic)

## Exit Criteria
- [ ] Структура проекта создана
- [ ] SQLite база работает
- [ ] arXiv коллектор собирает данные
- [ ] Хабр коллектор работает
- [ ] LDA тематика работает
- [ ] Сравнение тем работает
- [ ] Streamlit UI отображает результаты
- [ ] Тесты проходят

## MVP Scope
- 2 источника: arXiv + Хабр
- 100-500 статей для анализа
- 5-10 тем на источник
- Простой heatmap пересечений
