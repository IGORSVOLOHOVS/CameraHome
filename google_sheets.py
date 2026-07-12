import os
import json
import gspread
from google.oauth2.service_account import Credentials
from google.auth.transport.requests import AuthorizedSession
import config
import logger_db

# Google Sheets and Drive API Scopes
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
    Uploads a file to Google Drive under the Service Account and makes it readable by anyone.
    Returns: Direct download link for the image, or None if upload failed.
    """
    if not os.path.exists(config.CREDENTIALS_FILE):
        print("[ERROR] Credentials file not found, cannot upload to Drive.")
        return None

    try:
        creds = Credentials.from_service_account_file(config.CREDENTIALS_FILE, scopes=SCOPES)
        session = AuthorizedSession(creds)
        
        metadata = {
            'name': filename,
            'mimeType': 'image/jpeg'
        }
        
        files = {
            'data': ('metadata', json.dumps(metadata), 'application/json; charset=UTF-8'),
            'file': ('image', open(file_path, 'rb'), 'image/jpeg')
        }
        
        url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
        r = session.post(url, files=files, timeout=25)
        
        if r.status_code == 200:
            file_id = r.json().get('id')
            
            # Make the file public so Google Sheets =IMAGE() can read it
            permission_url = f"https://www.googleapis.com/drive/v3/files/{file_id}/permissions"
            session.post(permission_url, json={'role': 'reader', 'type': 'anyone'}, timeout=10)
            
            # Direct link to access image
            direct_link = f"https://docs.google.com/uc?export=download&id={file_id}"
            return direct_link
        else:
            print(f"[ERROR] Google Drive API upload failed ({r.status_code}): {r.text}")
            return None
    except Exception as e:
        print(f"[ERROR] Exception during Google Drive upload: {e}")
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
