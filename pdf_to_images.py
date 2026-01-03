import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import fitz  

DEFAULT_PDF = "ENGINEERING DRAWING BY N.D BHATT.pdf"
DEFAULT_OUT_DIR = "output_images"
DEFAULT_DPI = 600  # Increased to 600 DPI for highest OCR quality
DEFAULT_FMT = "png"
DEFAULT_WORKERS = 4  # Increased workers for better performance
DEFAULT_NO_ALPHA = True
START_PAGE = 400  # Start from page 400
END_PAGE = 500    # End at page 500

def render_page(pdf_path: str, page_number: int, out_path: str, dpi: int, image_format: str, alpha: bool):
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number)  # 0-based
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        if image_format.lower() in ("jpg", "jpeg"):
            alpha = False
        pix = page.get_pixmap(matrix=mat, alpha=alpha)
        pix.save(out_path)
        doc.close()
        return (page_number + 1, True, None)
    except Exception as e:
        return (page_number + 1, False, str(e))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def process_serial(tasks, pdf_path, dpi, image_format, alpha):
    for page_number, out_path in tasks:
        page, ok, err = render_page(pdf_path, page_number, out_path, dpi, image_format, alpha)
        if ok:
            print(f"Saved: {out_path}")
        else:
            print(f"Failed page {page}: {err}")


def process_parallel(tasks, pdf_path, dpi, image_format, alpha, max_workers):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for page_number, out_path in tasks:
            fut = executor.submit(render_page, pdf_path, page_number, out_path, dpi, image_format, alpha)
            futures[fut] = out_path

        completed = 0
        n_pages = len(tasks)
        for fut in as_completed(futures):
            completed += 1
            out_path = futures[fut]
            try:
                page, ok, err = fut.result()
                if ok:
                    print(f"[{completed}/{n_pages}] Saved page {page} -> {out_path}")
                else:
                    print(f"[{completed}/{n_pages}] FAILED page {page}: {err}")
            except Exception as e:
                print(f"[{completed}/{n_pages}] Worker exception: {e}")


def main():
    pdf_path = DEFAULT_PDF
    out_dir = DEFAULT_OUT_DIR
    dpi = max(72, int(DEFAULT_DPI))
    image_format = DEFAULT_FMT.lower()
    workers = max(1, int(DEFAULT_WORKERS))
    alpha = (not DEFAULT_NO_ALPHA) and (image_format == "png")
    start_page = START_PAGE
    end_page = END_PAGE

    if not os.path.isfile(pdf_path):
        raise SystemExit(f"Input PDF not found: {pdf_path}")

    ensure_dir(out_dir)

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    doc.close()

    # Validate page range
    if start_page < 1 or start_page > total_pages:
        raise SystemExit(f"Start page {start_page} is out of range (1-{total_pages})")
    if end_page < start_page or end_page > total_pages:
        raise SystemExit(f"End page {end_page} is invalid (must be between {start_page} and {total_pages})")

    n_pages = end_page - start_page + 1
    pad = max(4, len(str(end_page)))
    ext = "jpg" if image_format in ("jpg", "jpeg") else "png"

    print(f"PDF has {total_pages} total pages.")
    print(f"Processing pages {start_page}-{end_page} ({n_pages} pages)")
    print(f"Rendering at {dpi} DPI to {image_format.upper()} with {workers} worker(s) for optimal OCR quality.")
    print(f"Saving files to: {os.path.abspath(out_dir)} (pattern: image_{{page:{pad}d}}.{ext})")

    tasks = []
    for p in range(start_page - 1, end_page):
        filename = f"image_{p+1:0{pad}d}.{ext}"
        out_path = os.path.join(out_dir, filename)
        tasks.append((p, out_path))

    if workers == 1:
        process_serial(tasks, pdf_path, dpi, image_format, alpha)
    else:
        process_parallel(tasks, pdf_path, dpi, image_format, alpha, workers)

    print(f"Done. Processed {n_pages} pages (400-500).")


if __name__ == "__main__":
    main()