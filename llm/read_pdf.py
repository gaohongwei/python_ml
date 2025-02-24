import sys
import fitz  # PyMuPDF
import pyttsx3

FILE_PATH = './HandsOnLLM2024.pdf'

def read_pdf_with_voice(start_page, start_line):
    doc = fitz.open(FILE_PATH)
    engine = pyttsx3.init()

    if start_page < 0 or start_page >= len(doc):
        print(f"起始页码超出范围，PDF 共 {len(doc)} 页（从 0 开始计数）。")
        return

    # 将行号转换为索引（用户从 1 开始）
    line_num = start_line - 1
    page_num = start_page - 1

    while page_num < len(doc):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        lines = text.split("\n")

        if line_num >= len(lines):
            page_num += 1
            line_num = 0
            continue

        current_line = lines[line_num].strip()

        if not current_line:
            line_num += 1
            continue

        print(f"第 {page_num} 页，第 {line_num + 1} 行：{current_line}")
        engine.say(current_line)
        engine.runAndWait()

        line_num += 1

    print("📘 已朗读完整个 PDF 文档。")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法：python read_pdf.py <起始页码（从0开始）> <起始行号（从1开始）>")
        sys.exit(1)

    start_page = int(sys.argv[1])
    start_line = int(sys.argv[2])

    read_pdf_with_voice(start_page, start_line)
