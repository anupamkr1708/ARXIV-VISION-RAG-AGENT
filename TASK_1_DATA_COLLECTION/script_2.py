#!/usr/bin/env python3

import argparse
from pathlib import Path
import time
import pandas as pd
import arxiv
import fitz  # PyMuPDF for extracting full text from PDFs
from tqdm import tqdm

def extract_arxiv_id(url):
    """Extract arXiv ID from the given URL."""
    if isinstance(url, str):
        return url.split('/')[-1]
    return ""

def extract_text_from_pdf(pdf_path):
    """Extract full text from a downloaded PDF."""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + "\n"
        return full_text.strip()  # Remove extra spaces
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def main(input_csv, output_csv):
    """Reads a CSV file, fetches metadata from arXiv, and saves an updated CSV."""
    df = pd.read_csv(input_csv)

    # Create the PDFs directory
    pdf_dir = Path('arxiv_pdfs')
    pdf_dir.mkdir(exist_ok=True)

    # Ensure necessary columns exist
    if 'pdf_path' not in df.columns:
        df['pdf_path'] = ""
    if 'full_text' not in df.columns:  # New column for extracted text
        df['full_text'] = ""

    # Iterate through the dataset
    for index, row in tqdm(df.iterrows(), total=len(df)):
        client = arxiv.Client()
        arxiv_id = extract_arxiv_id(row['arXiv_link'])

        if not arxiv_id:  # Skip if no arXiv ID found
            continue

        try:
            # Fetch the paper details from arXiv
            search = arxiv.Search(id_list=[arxiv_id])
            first_result = next(client.results(search))

            # Define PDF path
            pdf_path = pdf_dir / f"{arxiv_id}.pdf"

            # Store extracted metadata
            df.at[index, 'arXiv_title'] = first_result.title
            df.at[index, 'abstract'] = first_result.summary  # Renamed from 'summary'
            df.at[index, 'publication_date'] = first_result.published.strftime("%Y-%m-%d")  # Extracted publication date

            # Download the PDF if it doesn't exist
            if not pdf_path.exists():
                first_result.download_pdf(dirpath=pdf_dir, filename=f"{arxiv_id}.pdf")
                df.at[index, 'pdf_path'] = str(pdf_path)

            # Extract full text from the PDF
            df.at[index, 'full_text'] = extract_text_from_pdf(pdf_path)

        except Exception as e:
            print(f"Error fetching data for arXiv ID {arxiv_id}: {e}")

        # Save progress every 10 iterations
        if index % 10 == 0:
            df.to_csv(output_csv, index=False)
            time.sleep(5)

    # Final save
    df.to_csv(output_csv, index=False)
    print("Data extraction complete and saved to", output_csv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download PDFs from arXiv and update CSV.")
    parser.add_argument('input_csv', help="Path to the input CSV file")
    parser.add_argument('output_csv', help="Path to the output CSV file")

    args = parser.parse_args()
    main(args.input_csv, args.output_csv)
