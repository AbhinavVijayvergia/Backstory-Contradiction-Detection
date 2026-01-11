import csv
import os

DATA_DIR = "data"
NOVELS_DIR = os.path.join(DATA_DIR, "novels")
CSV_PATH = os.path.join(DATA_DIR, "test.csv")

ROW_INDEX = 0


def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace(".txt", "")
    text = text.replace("the", "")
    text = text.replace(" ", "")
    return text


def find_novel_file(book_name: str) -> str:
    target = normalize(book_name)

    for fname in os.listdir(NOVELS_DIR):
        if normalize(fname) == target:
            return os.path.join(NOVELS_DIR, fname)

    raise FileNotFoundError(f"No novel file matches '{book_name}'")


def main():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    row = rows[ROW_INDEX]

    book_name = row["book_name"]
    char = row["char"]
    backstory = row["content"]

    novel_path = find_novel_file(book_name)

    with open(novel_path, "r", encoding="utf-8", errors="ignore") as f:
        novel_text = f.read()

    print("BOOK NAME :", book_name)
    print("CHARACTER :", char)
    print("NOVEL LENGTH     :", len(novel_text))
    print("BACKSTORY LENGTH :", len(backstory))


if __name__ == "__main__":
    main()
