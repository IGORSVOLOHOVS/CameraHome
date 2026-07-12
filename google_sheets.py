import os
import gspread
from google.oauth2.service_account import Credentials
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
        # Fetching first row
        existing_values = sheet.row_values(1)
        if not existing_values:
            sheet.append_row(["ID", "Timestamp", "Green Growth %", "Leaf/Plant Count", "Temperature", "Humidity"])

        rows_to_append = []
        ids_to_sync = []
        for row in unsynced:
            # row: (id, timestamp, green_percentage, leaf_count, temp, hum)
            ids_to_sync.append(row[0])
            rows_to_append.append([
                row[0], 
                row[1], 
                round(row[2], 2), 
                row[3], 
                row[4] if row[4] is not None else "", 
                row[5] if row[5] is not None else ""
            ])

        sheet.append_rows(rows_to_append)
        logger_db.mark_as_synced(ids_to_sync)
        print(f"[SUCCESS] Synced {len(rows_to_append)} rows to Google Sheets.")
        return True, len(rows_to_append)

    except Exception as e:
        print(f"[ERROR] Failed to sync to Google Sheets: {e}")
        return False, 0
