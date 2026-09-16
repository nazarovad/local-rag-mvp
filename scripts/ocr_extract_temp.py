from pathlib import Path
import json

from paddleocr import PPStructureV3


PDF_PATH = Path("data/Акт №91 от 28.04.2022.pdf")
OUTPUT_DIR = Path("data/ocr")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"Распознаём документ: {PDF_PATH}")

    pipeline = PPStructureV3(
        lang="ru",

        # Полезно для сканов:
        use_doc_orientation_classify=True,

        # Если страницы могут быть немного искривлены/сфотографированы:
        use_doc_unwarping=True,

        # Определяем ориентацию строк текста:
        use_textline_orientation=True,

        # Пока CPU — GPU занят Qwen
        device="cpu",
    )

    results = pipeline.predict(
        input=str(PDF_PATH)
    )

    markdown_pages = []
    all_json = []

    for page_no, result in enumerate(results, start=1):
        print(f"Обработана страница {page_no}")

        #
        # 1. Сохраняем исходный структурированный JSON
        #
        page_json_dir = OUTPUT_DIR / "json"
        page_json_dir.mkdir(exist_ok=True)

        result.save_to_json(
            save_path=str(page_json_dir)
        )

        #
        # 2. Сохраняем Markdown страницы
        #
        page_md_dir = OUTPUT_DIR / "pages"
        page_md_dir.mkdir(exist_ok=True)

        result.save_to_markdown(
            save_path=str(page_md_dir)
        )

        #
        # 3. Забираем Markdown в память
        #
        md_info = result.markdown

        markdown_pages.append(md_info)

        #
        # Для собственного отладочного JSON
        #
        try:
            all_json.append(result.json)
        except Exception:
            pass

    #
    # Объединяем Markdown всех страниц PDF
    #
    merged_markdown = pipeline.concatenate_markdown_pages(
        markdown_pages
    )

    markdown_path = OUTPUT_DIR / "document.md"

    markdown_path.write_text(
        merged_markdown,
        encoding="utf-8"
    )

    print()
    print("=" * 80)
    print("OCR завершён")
    print("=" * 80)

    print(f"Markdown: {markdown_path}")
    print(f"Страницы: {OUTPUT_DIR / 'pages'}")
    print(f"JSON: {OUTPUT_DIR / 'json'}")

    print()
    print("Первые 5000 символов:")
    print("=" * 80)
    print(merged_markdown[:5000])


if __name__ == "__main__":
    main()