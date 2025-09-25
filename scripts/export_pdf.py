import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright


SITE_DIR = Path(__file__).resolve().parents[1] / "site"
OUTPUT = SITE_DIR / "pdf" / "OilSimulatorDocs.pdf"

# Печатаем объединённую страницу, которую делает mkdocs-print-site-plugin
HTML_ENTRY = SITE_DIR / "print_page" / "index.html"


async def main():
    if not HTML_ENTRY.exists():
        print("Build the site first: mkdocs build", file=sys.stderr)
        sys.exit(1)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page()
        # Загружаем объединённую страницу печати
        await page.goto(HTML_ENTRY.as_uri(), wait_until="load")
        # Ждём загрузки шрифтов и переключаем media на print
        try:
            await page.evaluate("() => document.fonts && document.fonts.ready")
        except Exception:
            pass
        await page.emulate_media(media="print")
        # Ждём и запускаем MathJax
        await page.wait_for_function("() => window.MathJax && MathJax.typesetPromise", timeout=60000)
        await page.evaluate("() => MathJax.typesetPromise()")
        # Ждём отрисовки Mermaid (если есть блоки)
        await page.wait_for_function(
            "() => { const els = Array.from(document.querySelectorAll('.mermaid')); if (els.length===0) return true; return els.every(el => el.querySelector('svg')); }",
            timeout=60000,
        )
        # Небольшая задержка на компоновку
        await page.wait_for_timeout(500)
        await page.pdf(path=str(OUTPUT), format="A4", print_background=True, prefer_css_page_size=True)
        await browser.close()
        print(f"Saved PDF to {OUTPUT}")


if __name__ == "__main__":
    asyncio.run(main())


