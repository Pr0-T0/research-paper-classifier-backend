import fitz
import re


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def extract_paper_details(pdf_path):

    doc = fitz.open(pdf_path)

    num_pages = len(doc)

    full_text = ""

    for page in doc[:3]:
        full_text += page.get_text()

    full_text = clean_text(full_text)

    title = ""

    candidates = []

    # scan first 3 pages
    for page_num in range(min(3, len(doc))):

        page = doc[page_num]

        blocks = page.get_text("dict")["blocks"]

        for block in blocks:

            if "lines" not in block:
                continue

            block_text = ""
            font_sizes = []
            y_positions = []

            for line in block["lines"]:

                line_text = ""

                for span in line["spans"]:

                    text = span["text"].strip()

                    if not text:
                        continue

                    line_text += " " + text

                    font_sizes.append(span["size"])

                    y_positions.append(span["bbox"][1])

                block_text += line_text + " "

            block_text = clean_text(block_text)

            if not block_text:
                continue

            # ================= FILTERS =================

            if len(block_text.split()) < 4:
                continue

            if len(block_text) > 250:
                continue

            lower = block_text.lower()

            bad_words = [
                "abstract",
                "introduction",
                "keywords",
                "references",
                "conference",
                "journal",
                "proceedings",
                "arxiv",
                "ieee",
                "copyright"
            ]

            if any(word in lower for word in bad_words):
                continue

            if "@" in block_text:
                continue

            # average font size
            avg_size = sum(font_sizes) / len(font_sizes)

            # top position
            top_y = min(y_positions)

            candidates.append({
                "text": block_text,
                "size": avg_size,
                "y": top_y,
                "page": page_num
            })

    # sort:
    # 1. larger font
    # 2. closer to top
    candidates = sorted(
        candidates,
        key=lambda x: (-x["size"], x["y"])
    )

    if candidates:
        title = candidates[0]["text"]
    else:
        title = "Unknown Title"

    # =========================
    # ABSTRACT EXTRACTION
    # =========================

    abstract = ""

    lower_text = full_text.lower()

    start = lower_text.find("abstract")

    if start != -1:

        abstract_text = full_text[start + len("abstract"):]

        stoppers = [
            "keywords",
            "index terms",
            "i. introduction",
            "1 introduction",
            "1. introduction",
            "introduction"
        ]

        end_positions = []

        for stopper in stoppers:

            pos = abstract_text.lower().find(stopper)

            if pos != -1:
                end_positions.append(pos)

        if end_positions:

            end = min(end_positions)

            abstract = abstract_text[:end]

        else:
            # fallback length
            abstract = abstract_text[:2000]

    abstract = clean_text(abstract)

    # remove weird leading symbols
    abstract = re.sub(r"^[—:.\-\s]+", "", abstract)

    # =========================
    # AUTHORS EXTRACTION
    # =========================

    authors = "Unknown"

    try:

        first_page = doc[0]

        blocks = first_page.get_text("dict")["blocks"]

        title_bottom = 0

        # locate title position
        for block in blocks:

            if "lines" not in block:
                continue

            for line in block["lines"]:

                for span in line["spans"]:

                    if title[:25] in span["text"]:

                        title_bottom = max(
                            title_bottom,
                            span["bbox"][3]
                        )

        author_candidates = []

        for block in blocks:

            if "lines" not in block:
                continue

            for line in block["lines"]:

                for span in line["spans"]:

                    text = clean_text(span["text"])

                    y = span["bbox"][1]

                    if (
                        title_bottom < y < 350
                        and len(text.split()) <= 10
                        and "@" not in text
                        and "university" not in text.lower()
                        and "department" not in text.lower()
                        and "abstract" not in text.lower()
                        and "introduction" not in text.lower()
                    ):

                        author_candidates.append(text)

        # remove duplicates
        author_candidates = list(dict.fromkeys(author_candidates))

        if author_candidates:
            authors = ", ".join(author_candidates[:5])

    except:
        pass

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "pages": num_pages
    }