import os
import json
import gspread
import requests
from google.oauth2.service_account import Credentials
from google.auth.transport.requests import AuthorizedSession
import config
import logger_db

# Google Sheets API Scopes
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

def get_sheets_client():
    if not os.path.exists(config.CREDENTIALS_FILE):
        raise FileNotFoundError(f"Credentials file {config.CREDENTIALS_FILE} not found. Please place your service account JSON file here.")
    
    creds = Credentials.from_service_account_file(config.CREDENTIALS_FILE, scopes=SCOPES)
    return gspread.authorize(creds)

def upload_to_drive(file_path, filename):
    """
    Uploads a file to Catbox.moe and returns a public URL.
    This bypasses Google Drive Service Account quota limits completely.
    """
    try:
        from PIL import Image
        # Compress image to prevent network timeouts
        img = Image.open(file_path)
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        temp_upload_path = "temp_catbox_upload.jpg"
        img.save(temp_upload_path, "JPEG", quality=75)

        url = "https://catbox.moe/user/api.php"
        data = {
            "reqtype": "fileupload"
        }
        
        with open(temp_upload_path, "rb") as f:
            files = {"fileToUpload": f}
            r = requests.post(url, data=data, files=files, timeout=25)
            
        # Clean up temp file
        if os.path.exists(temp_upload_path):
            os.remove(temp_upload_path)
            
        if r.status_code == 200 and r.text.startswith("https://"):
            direct_link = r.text.strip()
            print(f"[SUCCESS] Uploaded photo to Catbox: {direct_link}")
            return direct_link
        else:
            print(f"[ERROR] Catbox upload failed ({r.status_code}): {r.text}")
            return None
    except Exception as e:
        print(f"[ERROR] Exception during Catbox upload: {e}")
        return None

def sync_to_sheets():
    """
    Fetches pending metrics from local database and appends them to Google Sheets.
    """
    if not config.SPREADSHEET_ID:
        print("[INFO] SPREADSHEET_ID not configured in .env. Skipping Google Sheets sync.")
        return False, 0

    unsynced = logger_db.get_unsynced_metrics()
    if not unsynced:
        print("[INFO] No pending metrics to sync.")
        return True, 0

    try:
        client = get_sheets_client()
        sheet = client.open_by_key(config.SPREADSHEET_ID).sheet1

        # Check if sheet is empty to write header
        existing_values = sheet.row_values(1)
        if not existing_values:
            sheet.append_row(["ID", "Timestamp", "Green Growth %", "Leaf/Plant Count", "Temperature", "Humidity", "Photo"])

        rows_to_append = []
        ids_to_sync = []
        for row in unsynced:
            # row: (id, timestamp, green_percentage, leaf_count, temp, hum, photo_url)
            ids_to_sync.append(row[0])
            
            photo_cell = ""
            if row[6]:
                # Combine HYPERLINK and IMAGE so it embeds and is clickable to view full size
                photo_cell = f'=HYPERLINK("{row[6]}", IMAGE("{row[6]}"))'

            rows_to_append.append([
                row[0], 
                row[1], 
                round(row[2], 2), 
                row[3], 
                row[4] if row[4] is not None else "", 
                row[5] if row[5] is not None else "",
                photo_cell
            ])

        sheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
        logger_db.mark_as_synced(ids_to_sync)
        print(f"[SUCCESS] Synced {len(rows_to_append)} rows to Google Sheets.")
        return True, len(rows_to_append)

    except Exception as e:
        print(f"[ERROR] Failed to sync to Google Sheets: {e}")
        return False, 0
