#!/usr/bin/env python3
"""
Экспорт документации в PDF через Playwright (Chromium).
Адаптирован для проекта oil-simulator-py-impes.
"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright


async def export_pdf() -> bool:
    project_root = Path(__file__).parent.parent
    print_page = project_root / "site" / "print_page" / "index.html"
    output_pdf = project_root / "site" / "pdf" / "OilSimulatorDocs.pdf"

    if not print_page.exists():
        print(f"❌ Ошибка: {print_page} не найден.")
        print("   Сначала выполните: mkdocs build")
        return False

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ЭКСПОРТ ДОКУМЕНТАЦИИ В PDF")
    print("=" * 70)
    print(f"\nВходной файл: {print_page}")
    print(f"Выходной PDF: {output_pdf}")

    async with async_playwright() as p:
        print("\n1. Запуск браузера Chromium...")
        browser = await p.chromium.launch()
        page = await browser.new_page()

        print("2. Загрузка документации...")
        await page.goto(f"file://{print_page.absolute()}", wait_until="networkidle")

        print("3. Ожидание рендеринга MathJax...")
        await page.wait_for_timeout(10_000)
        await page.evaluate(
            """
            async () => {
                if (!window.MathJax) {
                    return;
                }
                const mj = window.MathJax;
                if (typeof mj.typesetPromise === "function") {
                    await mj.typesetPromise();
                    console.log('MathJax v3 typeset complete');
                    return;
                }
                if (mj.Hub && typeof mj.Hub.Queue === "function") {
                    await new Promise((resolve) => {
                        mj.Hub.Queue(["Typeset", mj.Hub, () => {
                            console.log('MathJax v2 typeset complete');
                            resolve();
                        }]);
                    });
                } else {
                    console.warn("MathJax detected, but no typeset API available.");
                }
            }
        """
        )
        await page.wait_for_timeout(10_000)

        math_count = await page.evaluate(
            """
            () => {
                const arithmatex = document.querySelectorAll('.arithmatex').length;
                const mjx = document.querySelectorAll('mjx-container').length;
                return { arithmatex, mjx };
            }
        """
        )
        print(
            f"   ✓ MathJax: {math_count['arithmatex']} arithmatex блоков, {math_count['mjx']} mjx элементов"
        )

        print("4. Проверка Mermaid диаграмм...")
        selectors = [
            ".mermaid svg",
            'svg[id^="mermaid"]',
            'svg[class*="mermaid"]',
            ".language-mermaid",
            "pre.mermaid",
        ]
        found = False
        for selector in selectors:
            count = await page.locator(selector).count()
            if count > 0:
                print(f"   ✓ Найдено элементов по селектору '{selector}': {count}")
                found = True
                break
        if not found:
            print("   ℹ Mermaid диаграмм в явном виде не найдено (нормально, если не используются).")

        print("   Финальное ожидание...")
        await page.wait_for_timeout(10_000)

        print("5. Генерация PDF...")
        await page.pdf(
            path=str(output_pdf),
            format="A4",
            print_background=True,
            margin={"top": "20mm", "right": "15mm", "bottom": "20mm", "left": "15mm"},
            display_header_footer=True,
            header_template=(
                '<div style="font-size:9pt; width:100%; text-align:center; color:#666;">'
                "Oil Simulator — документация</div>"
            ),
            footer_template=(
                '<div style="font-size:9pt; width:100%; text-align:center; color:#666;">'
                "<span class=\"pageNumber\"></span> / <span class=\"totalPages\"></span></div>"
            ),
        )

        await browser.close()

    print(f"\n✅ PDF успешно создан: {output_pdf}")
    print(f"   Размер: {output_pdf.stat().st_size / 1024:.1f} КБ")
    return True


def main() -> None:
    site_dir = Path(__file__).parent.parent / "site"
    if not site_dir.exists():
        print("❌ Директория site/ не найдена.")
        print("   Сначала выполните: mkdocs build")
        raise SystemExit(1)

    success = asyncio.run(export_pdf())
    if not success:
        raise SystemExit(1)

    print("\n" + "=" * 70)
    print("ГОТОВО! PDF документация создана.")
    print("=" * 70)
    print("\nКоманды для открытия:")
    print("  xdg-open site/pdf/OilSimulatorDocs.pdf  # Linux")
    print("  open site/pdf/OilSimulatorDocs.pdf      # macOS")
    print("  start site/pdf/OilSimulatorDocs.pdf     # Windows")


if __name__ == "__main__":
    main()

