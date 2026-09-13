from app.services.ingestion import load_file

from pathlib import Path
docs=load_file(Path("D:\Projects\Enterprise-IT-Support\data\sample_kb\company_it_handbook.md"))

print(docs[0].page_content)
