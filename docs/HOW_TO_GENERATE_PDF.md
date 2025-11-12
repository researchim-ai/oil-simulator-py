# Генерация документации и экспорт в PDF

## 1. Установка зависимостей

```bash
pip install -r requirements-docs.txt
playwright install chromium
```

## 2. Структура документации

- Основные Markdown-файлы находятся в `docs/`.
- Конфигурация MkDocs — `mkdocs.yml`.
- Дополнительные ресурсы: `docs/javascripts/`, `docs/stylesheets/`.

## 3. Генерация HTML-версии

```bash
mkdocs build
```

После успешного выполнения статический сайт размещается в директории `site/`.

## 4. PDF через плагин Print Site

При сборке MkDocs создаёт отдельную страницу `site/print_page/index.html`, содержащую объединённую версию документации.

## 5. Экспорт PDF

```bash
python scripts/export_pdf.py
```

Скрипт делает следующее:

1. Проверяет наличие `site/print_page/index.html`.
2. Запускает браузер Chromium через Playwright.
3. Ждёт прогрузку MathJax и Mermaid diagram.
4. Сохраняет PDF в `site/pdf/OilSimulatorDocs.pdf`.

Настройки (формат A4, поля 20/15 мм) можно изменить в `scripts/export_pdf.py`.

## 6. Просмотр результатов

- HTML: откройте `site/index.html` в браузере.
- PDF: `site/pdf/OilSimulatorDocs.pdf`.

## 7. Автоматизация

Для автоматического обновления документации можно добавить Makefile:

```Makefile
.PHONY: docs pdf

docs:
	mkdocs build

pdf: docs
	python scripts/export_pdf.py
```

## 8. Распространённые ошибки

- **Chromium не установлен:** выполните `playwright install chromium`.
- **MathJax не подгрузился:** убедитесь, что интернет доступен (подгружаются CDN-скрипты) или разместите их локально.
- **Mermaid не отрисовывается:** проверьте, что блоки объявлены как ```` ```mermaid ```` и включён плагин `mkdocs-mermaid2-plugin`.

## 9. Обновление версии

Измените имя файла PDF в `mkdocs.yml` (`path_to_pdf`) и в `scripts/export_pdf.py`, если требуется выпуск версионных сборок.

