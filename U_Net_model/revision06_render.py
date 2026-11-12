"""Render revision-06 DOCX with installed Microsoft Word for page inspection."""

import sys
from pathlib import Path

from pdf2image import convert_from_path
from win32com.client import DispatchEx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import SubprojectPaths

DELIVERY = (Path.home() / "OneDrive/Desktop/文献与汇报/02_自研课题与论文/"
            "FiLM调制UNet降水预测/修改1/06_对照研究返修")
OUTPUT = SubprojectPaths(__file__).results_dir / "revision06_docs_qa"
NAMES = ("06_对照研究稿", "06_Response_EN", "06_审稿回复中文",
         "Highlights", "06_Supplement_S1")


def main():
    app = DispatchEx("Word.Application")
    app.Visible = False
    app.DisplayAlerts = 0
    try:
        for name in NAMES:
            source = DELIVERY / f"{name}.docx"
            target = OUTPUT / name
            target.mkdir(parents=True, exist_ok=True)
            pdf = target / f"{name}.pdf"
            document = app.Documents.Open(str(source), ReadOnly=True)
            try:
                document.ExportAsFixedFormat(str(pdf), 17)
            finally:
                document.Close(False)
            pages = convert_from_path(str(pdf), dpi=140)
            for i, page in enumerate(pages, 1):
                page.save(target / f"page-{i}.png")
            print(f"{name}: {len(pages)} pages")
    finally:
        app.Quit()


if __name__ == "__main__":
    main()
