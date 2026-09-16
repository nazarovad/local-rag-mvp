# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import shutil

from paddleocr import PPStructureV3


DOCUMENTS_DIR = Path("data/documents")


def main():
    parser = argparse.ArgumentParser(
        description="Распознавание PDF через PaddleOCR"
    )

    parser.add_argument(
        "pdf",
        help="Путь к PDF-файлу",
    )

    args = parser.parse_args()

    # ------------------------------------------------------
    # PDF
    # ------------------------------------------------------

    pdf_path = Path(args.pdf)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF не найден: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Ожидался PDF-файл: {pdf_path}"
        )

    # Имя документа = имя PDF без расширения
    document_name = pdf_path.stem

    # data/documents/ИМЯ_ФАЙЛА
    output_dir = (
        DOCUMENTS_DIR /
        document_name
    )

    # ------------------------------------------------------
    # Если документ уже существует —
    # удаляем старую папку целиком
    # ------------------------------------------------------

    if output_dir.exists():
        print(
            f"Папка уже существует. "
            f"Удаляем: {output_dir}"
        )

        shutil.rmtree(output_dir)

    # ------------------------------------------------------
    # Каталоги документа
    # ------------------------------------------------------

    # data/documents/ИМЯ/pages
    page_md_dir = (
        output_dir /
        "pages"
    )

    # data/documents/ИМЯ/pages/imgs
    imgs_dir = (
        page_md_dir /
        "imgs"
    )

    # data/documents/ИМЯ/json
    page_json_dir = (
        output_dir /
        "json"
    )

    page_md_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    imgs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    page_json_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(f"Распознаём документ: {pdf_path}")
    print(f"Папка документа:     {output_dir}")
    print("=" * 80)

    # ------------------------------------------------------
    # PaddleOCR
    # ------------------------------------------------------

    pipeline = PPStructureV3(
        lang="ru",

        use_doc_orientation_classify=True,

        use_doc_unwarping=True,

        use_textline_orientation=True,

        # GPU пока занят Qwen
        device="cpu",
    )

    results = pipeline.predict(
        input=str(pdf_path)
    )

    page_count = 0
    image_count = 0

    # ------------------------------------------------------
    # Обрабатываем страницы
    # ------------------------------------------------------

    for page_no, result in enumerate(
        results,
        start=1,
    ):
        page_count = page_no

        print(
            f"Обработана страница {page_no}"
        )

        # --------------------------------------------------
        # JSON
        # --------------------------------------------------

        result.save_to_json(
            save_path=str(
                page_json_dir
            )
        )

        # --------------------------------------------------
        # Markdown
        # PaddleOCR сохраняет Markdown в pages
        # --------------------------------------------------

        result.save_to_markdown(
            save_path=str(
                page_md_dir
            )
        )

        # --------------------------------------------------
        # Images
        # --------------------------------------------------

        md_info = result.markdown

        if isinstance(md_info, dict):
            markdown_images = md_info.get(
                "markdown_images",
                {}
            )

            for image_path, image in (
                markdown_images.items()
            ):
                relative_path = Path(
                    image_path
                )

                # Обычно PaddleOCR возвращает:
                #
                # imgs/xxx.png
                #
                # Поэтому:
                #
                # pages / imgs/xxx.png
                #
                # ->
                #
                # pages/imgs/xxx.png

                save_path = (
                    page_md_dir /
                    relative_path
                )

                save_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                image.save(
                    save_path
                )

                image_count += 1

    # ------------------------------------------------------
    # Result
    # ------------------------------------------------------

    print()
    print("=" * 80)
    print("OCR завершён")
    print("=" * 80)

    print(
        f"Документ:    {document_name}"
    )

    print(
        f"Страниц:     {page_count}"
    )

    print(
        f"Изображений: {image_count}"
    )

    print(
        f"Папка:       {output_dir}"
    )

    print(
        f"Markdown:    {page_md_dir}"
    )

    print(
        f"Images:      {imgs_dir}"
    )

    print(
        f"JSON:        {page_json_dir}"
    )


if __name__ == "__main__":
    main()