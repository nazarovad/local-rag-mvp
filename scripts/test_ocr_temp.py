from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
)


PDF_PATH = Path("data/акт №87 от 31.03.20.pdf")


pipeline_options = PdfPipelineOptions()

pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True

pipeline_options.ocr_options = EasyOcrOptions(
    lang=["ru", "en"],
    force_full_page_ocr=True,
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)

print(f"Распознаю: {PDF_PATH}")

result = converter.convert(PDF_PATH)

markdown = result.document.export_to_markdown()

Path("data/parsed-ocr.md").write_text(
    markdown,
    encoding="utf-8"
)

print(markdown[:5000])
print()
print("Результат сохранён в data/parsed-ocr.md")