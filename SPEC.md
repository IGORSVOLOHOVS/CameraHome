# CameraHome: software requirements specification

Reverse-engineered from the code at commit `f6f2902` on 2026-11-18. Every requirement describes behaviour that exists in the code and cites it. Behaviour promised elsewhere but missing from the code is listed under 3.8. The documentation hub is [docs/README.md](docs/README.md).

<!-- jira-sync:start -->
Jira: KAN-122

## Tasks

This list is one ticket in Jira project KAN, [KAN-122](https://igorsvolohovs.atlassian.net/browse/KAN-122). Each line is a checkbox there: tick it in Jira and the sync ticks it here; add a checkbox there and it is appended here. Done items are the features described in section 3; open items are what the project declares but the code does not do yet.

- [x] Configuration (section 3.1) <!-- spec-configuration -->
- [x] Startup and scheduling (`main.py`) (section 3.2) <!-- spec-startup-and-scheduling-main-py -->
- [x] Capture and analysis (`cv_yolo.py`) (section 3.3) <!-- spec-capture-and-analysis-cvyolo-py -->
- [x] Storage, hosting and sync (section 3.4) <!-- spec-storage-hosting-and-sync -->
- [x] Telegram bot (`telegram_bot.py`) (section 3.5) <!-- spec-telegram-bot-telegrambot-py -->
- [x] Person detector (`vision_mqtt.py`) (section 3.6) <!-- spec-person-detector-visionmqtt-py -->
- [x] Reference to the shared standard (section 3.7) <!-- spec-reference-to-the-shared-standard -->
- [ ] `/status` reports "last measured metrics" (declared, not implemented) <!-- spec-status-reports-last-measured-metrics -->
- [ ] Leaving `TELEGRAM_CHAT_ID` as `your_telegram_chat_id` triggers auto-discovery (declared, not implemented) <!-- spec-leaving-telegramchatid-as-yourtelegramchatid-trigg -->
- [ ] Temperature and humidity "in future, read from MQTT" (declared, not implemented) <!-- spec-temperature-and-humidity-in-future-read-from-mqtt -->
- [ ] MQTT for the plant monitor "for future expansion" (declared, not implemented) <!-- spec-mqtt-for-the-plant-monitor-for-future-expansion -->
- [ ] Branch `plant-growth-metrics-termux` (declared, not implemented) <!-- spec-branch-plant-growth-metrics-termux -->
- [ ] Upload "to Google Drive" (declared, not implemented) <!-- spec-upload-to-google-drive -->

<!-- jira-sync:end -->

## 1. Purpose and scope

**In scope**
- Plant growth monitor (`main.py` and the modules it imports): scheduled camera capture on an Android phone under Termux, green-area and YOLO analysis, SQLite cache, Google Sheets sync, photo hosting on catbox.moe, a Telegram bot for control and reports, and self-update from git.
- Person detector (`vision_mqtt.py`): a capture-and-detect loop with Telegram and MQTT alerts.

**Out of scope**
- Training or conversion of the YOLO model. The repository ships only `yolov8n_float16.tflite`, and `.gitignore:1-5` excludes the other model artefacts.
- Temperature and humidity measurement (not implemented, see 3.8).
- Termux, Termux:API, Google Cloud and BotFather setup, beyond what `README.md:29-105` describes.

## 2. Context

| Actor or system | Role | Source |
|---|---|---|
| Bot owner (Telegram chat) | Receives reports and sends commands. One chat ID | `telegram_bot.py:11-13`, `:168` |
| Termux:API `termux-camera-photo` | Takes the photo | `cv_yolo.py:31`, `vision_mqtt.py:118` |
| TFLite runtime (`tflite_runtime`, falling back to `tensorflow.lite`) | Runs YOLO on the device | `cv_yolo.py:8-11`, `vision_mqtt.py:23-26` |
| Telegram Bot API | `getUpdates`, `sendMessage`, `sendPhoto` | `telegram_bot.py:16-60`, `vision_mqtt.py:35-80` |
| Google Sheets API via gspread, service account | Measurement log | `google_sheets.py:11-21`, `:73-103` |
| catbox.moe | Anonymous photo hosting for the sheet previews | `google_sheets.py:36-43` |
| git remote | Source of `/update` | `updater.py:12-17` |
| MQTT broker | Person alerts (`vision_mqtt.py` only) | `vision_mqtt.py:97-109`, `:201` |

Runtime environment: Android with Termux and Termux:API (`README.md:31-33`). Python, with no version stated anywhere. The process is started from the repository root, because all paths are relative (`config.py:3`, `logger_db.py:4`, `cv_yolo.py:27`).

## 3. Functional requirements

The repository has no tests, so every requirement has the verification "no test". Where a requirement was checked offline while writing this spec, that is noted in brackets (see `docs/README.md`, Testing).

### 3.1 Configuration

- **FR-1.** The system shall read `.env` from the working directory, skip lines that start with `#` or contain no `=`, and store each `key=value` in the process environment, overwriting existing values. Source: `config.py:3-14`. Verification: no test (checked offline).
- **FR-2.** The system shall use these defaults when variables are unset: `CREDENTIALS_FILE=credentials.json`, `CAMERA_ID=0`, `MODEL_PATH=yolov8n_float16.tflite`, `THRESHOLD=0.25`, `INTERVAL_HOURS=3`. `SIMULATION` shall be true only for `true`, `1` or `yes` (any case). Source: `config.py:20-30`. Verification: no test.

### 3.2 Startup and scheduling (`main.py`)

- **FR-3.** The system shall exit with code 1 when `TELEGRAM_TOKEN` is not set. Source: `main.py:85-87`. Verification: no test.
- **FR-4.** At startup the system shall create the SQLite table, load the analyser, start the Telegram bot thread and send a startup message. Source: `main.py:12`, `:14`, `:89-99`. Verification: no test.
- **FR-5.** The system shall check once per second whether the interval has passed since the last successful measurement, and if so run one. The interval is 60 s in simulation mode and `INTERVAL_HOURS * 3600` s otherwise. The first check after start always fires. Source: `main.py:16`, `:103-114`. Verification: no test.
- **FR-6.** On `KeyboardInterrupt` the system shall stop the bot thread (3 s join timeout) and exit. Source: `main.py:115-120`, `telegram_bot.py:179-182`. Verification: no test.

### 3.3 Capture and analysis (`cv_yolo.py`)

- **FR-7.** The system shall delete any previous output file and run `termux-camera-photo -c <CAMERA_ID> <path>` with a 15 s timeout. Capture counts as successful only if the file exists and is not empty. Source: `cv_yolo.py:27-40`. Verification: no test.
- **FR-8.** When capture fails during a scheduled measurement, the system shall send a warning to Telegram and skip the rest of that measurement. Source: `main.py:24-30`. Verification: no test (checked offline).
- **FR-9.** The system shall compute the green percentage as 100 × (pixels with HSV in [35,40,40]..[85,255,255]) / (all pixels). Source: `cv_yolo.py:54-64`. Verification: no test.
- **FR-10.** The system shall draw the external contours of the green mask whose area is over 50 px, in green (BGR 0,255,0) with thickness 3. Source: `cv_yolo.py:66-69`. Verification: no test.
- **FR-11.** When the model is loaded, the system shall feed it the image as a 640×640 RGB float32 tensor scaled to 0..1 and take the scores of class 58 (class 0 if the model has 58 classes or fewer). It shall keep the boxes whose score exceeds `THRESHOLD`, apply NMS with IoU 0.45, report the number of kept boxes as the leaf/plant count, and draw each box in blue with the label `Plant: <conf>`. Source: `cv_yolo.py:74-133`. Verification: no test.
- **FR-12.** If the model file is missing or fails to load, the system shall skip detection and report a count of 0. If inference raises, the system shall log the error and return the green percentage with the count reached so far (0 unless the error happens while drawing boxes). Source: `cv_yolo.py:17-25`, `:71-74`, `:125`, `:135-136`. Verification: no test.
- **FR-13.** The system shall overwrite the input image file with the annotated image. If the image cannot be read, it shall return `(0.0, 0)`. Source: `cv_yolo.py:49-52`, `:139`. Verification: no test.

### 3.4 Storage, hosting and sync

- **FR-14.** The system shall store each scheduled measurement as a `metrics` row with the device-local timestamp `YYYY-MM-DD HH:MM:SS`, the green percentage, the count, empty temperature and humidity, the photo URL and `sync_status='pending'`. Source: `logger_db.py:29-40`, `main.py:40-46`. Verification: no test (checked offline).
- **FR-15.** The system shall add the `photo_url` column to an existing `metrics` table that lacks it. Source: `logger_db.py:21-25`. Verification: no test.
- **FR-16.** The system shall shrink the annotated photo to at most 1024×1024 px (JPEG quality 75), POST it to `https://catbox.moe/user/api.php` with `reqtype=fileupload` (25 s timeout), and use the response body as the URL only if the status is 200 and the body starts with `https://`. Otherwise the URL is empty. Source: `google_sheets.py:23-58`. Verification: no test.
- **FR-17.** The system shall append all `pending` rows to the first worksheet of `SPREADSHEET_ID`, using the service account in `CREDENTIALS_FILE`. It writes the header row first if row 1 is empty, enters values as `USER_ENTERED` with the photo cell `=HYPERLINK("<url>", IMAGE("<url>"))`, and then marks the rows `synced`. Source: `google_sheets.py:60-106`, `logger_db.py:42-60`. Verification: no test (checked offline with a stub client).
- **FR-18.** The system shall skip the sync when `SPREADSHEET_ID` is unset, and leave the rows `pending` when the sync raises, so the next measurement retries them. Source: `google_sheets.py:64-66`, `:108-110`. Verification: no test (checked offline).
- **FR-19.** After a scheduled measurement the system shall send the annotated photo with a caption (time, green %, count) to Telegram, and a message with the number of synced rows if that number is above 0. Source: `main.py:48-66`. Verification: no test.

### 3.5 Telegram bot (`telegram_bot.py`)

- **FR-20.** The system shall long-poll `getUpdates` (offset, `timeout=5`, 10 s request timeout) in a daemon thread, with a 1 s pause between polls (2 s while waiting for the first chat, FR-21). Source: `telegram_bot.py:52-60`, `:159`, `:161-177`. Verification: no test.
- **FR-21.** When no chat ID is configured (empty value), the system shall bind to the chat of the first incoming message and send a greeting. Source: `telegram_bot.py:147-158`. Verification: no test.
- **FR-22.** The system shall act only on text messages from the configured chat, and shall silently skip sending while no chat ID is known. Source: `telegram_bot.py:22-23`, `:30-31`, `:165-169`. Verification: no test.
- **FR-23.** The system shall shrink photos to at most 1024×1024 px (JPEG quality 75) before `sendPhoto` (25 s timeout). Source: `telegram_bot.py:29-50`. Verification: no test.
- **FR-24.** The system shall match commands case-insensitively by prefix and handle them as follows (source `telegram_bot.py:62-142`, verification: no test):
  - `/start`, `/help`: send the command list;
  - `/photo`: capture, analyse and send the annotated photo with results, or a failure message. Nothing is stored or synced;
  - `/status`: report the mode, the number of `pending` rows, the model file name and the camera ID (checked offline);
  - `/simulate`: toggle `config.SIMULATION` in memory and reset the scheduler;
  - `/interval <h>`: set a positive float number of hours in memory and reset the scheduler. A non-positive or non-numeric value gets an error message; no argument gets the current value;
  - `/sheets`: send the spreadsheet URL, or an error if `SPREADSHEET_ID` is unset;
  - `/update`: see FR-26;
  - anything else: a hint to use `/help`.
- **FR-25.** Resetting the scheduler shall set the last measurement time to 0, so the next loop iteration measures. Source: `main.py:71-75`. Verification: no test.
- **FR-26.** On `/update` the system shall run `git fetch` and `git status -uno`. If the status output contains "Your branch is behind" or "can be fast-forwarded", it runs `git pull`, reports the output and replaces the process via `os.execv` with the same interpreter and arguments. Otherwise it reports "Already up to date." or the git error. Source: `updater.py:5-32`, `telegram_bot.py:133-140`. Verification: no test.

### 3.6 Person detector (`vision_mqtt.py`)

- **FR-27.** The system shall take `--model`, `--host`, `--token`, `--chat_id`, `--threshold`, `--camera` and `--cooldown`, defaulting to `MODEL_PATH`, `MQTT_HOST`, `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, `THRESHOLD` (0.6), `CAMERA_ID` (0) and `COOLDOWN` (30). Source: `vision_mqtt.py:224-233`. Verification: no test.
- **FR-28.** The system shall connect to the MQTT broker on port 1883 when a host is given, and continue without MQTT when none is given or the connection fails. Source: `vision_mqtt.py:97-109`. Verification: no test.
- **FR-29.** In a loop with a 1 s pause, the system shall capture a frame (15 s timeout), run the model on a 640×640 input and take the highest class-0 score as the person confidence. Source: `vision_mqtt.py:111-147`, `:188-221`. Verification: no test.
- **FR-30.** When the confidence exceeds 0.3, the system shall draw a green box (thickness 20) with the label `PERSON <conf>` on the snapshot. Source: `vision_mqtt.py:150-178`. Verification: no test.
- **FR-31.** When the confidence exceeds the threshold and more than `cooldown` seconds have passed since the last alert, the system shall publish `{"status":"detected","confidence":<float>}` to `home/camera/detection`, and send the snapshot to Telegram shrunk to at most 640 px (JPEG quality 70). If no chat ID is known, it first takes the chat of the last update. Source: `vision_mqtt.py:37-55`, `:192-219`. Verification: no test.

### 3.7 Reference to the shared standard

Not applicable: the repository does not carry the repo-quality-template tooling (no `recipe.ps1`, `.claude/`, `.gitea/` or `.pre-commit-config.yaml` in `git ls-files`).

### 3.8 Declared but not implemented

| Claim | Where declared | What the code does |
|---|---|---|
| `/status` reports "last measured metrics" | `README.md:21` | Only the mode, the unsynced count, the model and the camera (`telegram_bot.py:90-101`) |
| Leaving `TELEGRAM_CHAT_ID` as `your_telegram_chat_id` triggers auto-discovery | `README.md:78` | Auto-discovery needs an empty value. The placeholder is used as a chat ID (`telegram_bot.py:147`, `:168`) |
| Temperature and humidity "in future, read from MQTT" | `main.py:43-44` | Always `None`. The sheet columns stay empty |
| MQTT for the plant monitor "for future expansion" | `config.py:32-35` | `MQTT_HOST`, `MQTT_PORT`, `MQTT_TOPIC` are read and never used |
| Branch `plant-growth-metrics-termux` | `README.md:66` | Not present in this clone, which has only `main` |
| Upload "to Google Drive" | `main.py:34-36`, name `upload_to_drive` | Uploads to catbox.moe (`google_sheets.py:23-27`) |

## 4. Interfaces

**CLI**

| Command | Arguments | Source |
|---|---|---|
| `python main.py` | none | `main.py:122-123`, `README.md:87` |
| `python vision_mqtt.py` | `--model PATH`, `--host HOST`, `--token TOKEN`, `--chat_id ID`, `--threshold FLOAT`, `--camera INT`, `--cooldown INT` | `vision_mqtt.py:224-231` |

**Telegram commands**: see FR-24. Autocomplete list: `README.md:98-105`.

**External calls**

| Target | Request | Timeout | Source |
|---|---|---|---|
| `api.telegram.org/bot<token>/getUpdates` | GET `offset`, `timeout=5` | 10 s | `telegram_bot.py:54-55` |
| `.../sendMessage` | POST `chat_id`, `text` (no `parse_mode`) | 10 s | `telegram_bot.py:25` |
| `.../sendPhoto` | POST `chat_id`, `caption`, file `photo` | 25 s (monitor), 10 s (detector) | `telegram_bot.py:41-43`, `vision_mqtt.py:72-74` |
| `catbox.moe/user/api.php` | POST `reqtype=fileupload`, file `fileToUpload` | 25 s | `google_sheets.py:36-43` |
| Google Sheets (gspread) | `open_by_key().sheet1`, `row_values(1)`, `append_row`, `append_rows(USER_ENTERED)`; scopes `spreadsheets`, `drive` | library default | `google_sheets.py:11-21`, `:75-103` |
| MQTT | topic `home/camera/detection`, JSON `{"status":"detected","confidence":float}` | keepalive 60 s | `vision_mqtt.py:93`, `:104`, `:201` |
| `termux-camera-photo` | `-c <id> <path>` | 15 s | `cv_yolo.py:31-32`, `vision_mqtt.py:118-120` |
| `git` | `fetch`; `status -uno`; `pull` | none | `updater.py:12-17` |

## 5. Data

**SQLite** `plant_metrics.db` in the working directory, table `metrics` (`logger_db.py:4-20`):

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | Also written to the sheet's `ID` column |
| `timestamp` | TEXT NOT NULL | Device local time, `%Y-%m-%d %H:%M:%S` |
| `green_percentage` | REAL NOT NULL | 0..100 |
| `leaf_count` | INTEGER NOT NULL | Number of class-58 detections after NMS |
| `temperature`, `humidity` | REAL | Always NULL at present |
| `photo_url` | TEXT | catbox.moe URL or NULL |
| `sync_status` | TEXT DEFAULT `pending` | `pending` or `synced` |

**Google Sheet**, first worksheet. Header: `ID, Timestamp, Green Growth %, Leaf/Plant Count, Temperature, Humidity, Photo`. The percentage is rounded to 2 decimals and the photo cell is a formula (`google_sheets.py:80`, `:88-101`).

**Working files**, all relative to the working directory: `snap.jpg` (capture, then the annotated image), `snap_compressed.jpg` (Telegram copy), `temp_catbox_upload.jpg` (upload copy), `snap_tg.jpg` (detector copy). Source: `cv_yolo.py:27`, `telegram_bot.py:37`, `google_sheets.py:33`, `vision_mqtt.py:94`, `:213`.

**Secrets**: `.env` (bot token, chat ID, spreadsheet ID) and `credentials.json` (service account key). Both are git-ignored (`.gitignore:6`, `:10`).

## 6. Non-functional requirements

- **NFR-1.** Inference shall run on the device through a TFLite interpreter, not a remote service. Source: `cv_yolo.py:8-20`, `vision_mqtt.py:23-26`, `:88-89`.
- **NFR-2.** Every network and camera call shall have a bounded timeout (values in section 4). The exceptions are `git`, which has none, and the Google Sheets calls, for which the code sets no timeout and leaves the gspread default (not checked). Source: section 4 rows.
- **NFR-3.** Images shall be shrunk before network transfer: 1024 px / JPEG 75 (monitor), 640 px / JPEG 70 (detector). Source: `telegram_bot.py:36-38`, `google_sheets.py:32-34`, `vision_mqtt.py:212-213`.
- **NFR-4.** A failed sync shall not lose measurements: they stay `pending` in SQLite until a later sync succeeds. Source: `google_sheets.py:68-110`, `logger_db.py:42-60`.
- **NFR-5.** Bot commands shall be accepted only from the single configured or discovered chat. Source: `telegram_bot.py:168`.
- **NFR-6.** Alerts in the detector shall be limited to one per `cooldown` seconds. Source: `vision_mqtt.py:198`, `:219`.
- **NFR-7.** Credentials shall be kept out of version control. Source: `.gitignore:6`, `:10`.

## 7. Constraints and dependencies

There is no lock file or manifest, and no version is pinned. The list comes from the imports and from `README.md:54`, `:60`.

| Dependency | Used by | Installed by |
|---|---|---|
| Python (version unstated) | all | `pkg install python` (`README.md:54`) |
| `numpy` | `cv_yolo.py`, `vision_mqtt.py` | `pkg install python-numpy` |
| OpenCV `cv2` (uses `cv2.dnn.NMSBoxes`) | `cv_yolo.py`, `vision_mqtt.py` | `pkg install opencv-python` |
| Pillow (uses `Image.Resampling`; inference: Pillow 9.1 or later) | `telegram_bot.py`, `google_sheets.py`, `vision_mqtt.py` | `pkg install python-pillow` |
| `tflite-runtime`, or `tensorflow` as fallback | `cv_yolo.py`, `vision_mqtt.py` | `pip install tflite-runtime` |
| `requests` | Telegram, catbox.moe | `pip install requests` |
| `gspread`, `google-auth` | `google_sheets.py` | `pip install gspread google-auth` |
| `paho-mqtt` | `vision_mqtt.py` | not in the current README |
| `termux-api`, `git` | capture, `/update` | `pkg install termux-api ... git` |

Model: `yolov8n_float16.tflite`. Inference from the comments at `cv_yolo.py:87` and `vision_mqtt.py:144`: the output is `[1, 84, 8400]`, that is 4 box values plus 80 COCO classes. The `metadata.json` embedded in the model file agrees: 80 COCO names, `imgsz` 640×640, `nms: false` (see open question 3).

## 8. Acceptance

There are no automated tests. On a Termux device configured per `README.md`, with `TELEGRAM_CHAT_ID` left empty:

1. `python main.py` starts. The startup message is not delivered, because no chat ID is known yet (FR-22). The first message from the owner links the chat and gets the greeting (FR-3, FR-4, FR-21).
2. A measurement runs immediately. A row appears in `plant_metrics.db`, and with a valid `SPREADSHEET_ID` the sheet gets a header and a row with an image preview. Telegram receives the annotated photo with green % and count only if the chat was linked before the photo is sent; otherwise use `/simulate` or wait for the next measurement (FR-5, FR-7 to FR-19, FR-22).
3. With the network off, a measurement leaves the row `pending`. After the network is back, the next measurement syncs it (FR-18, NFR-4).
4. Each command in FR-24 gives the listed reply. `/interval 0` and `/interval x` give the error message.
5. `/update` on a branch that is behind pulls and restarts (FR-26).
6. `python vision_mqtt.py --token ... --host ...`, with a person in frame, gives one MQTT message and one Telegram photo per cooldown (FR-27 to FR-31).

The offline checks done for this spec covered FR-1, FR-8, FR-14, FR-17, FR-18 and `/status` only. An independent re-run on 2026-11-18 (stubbed `cv2`, `tflite_runtime` and `gspread`, in a temporary directory) gave the same results, including the `AttributeError` from a failing `git pull`, and showed that `send_message` sends nothing while the chat ID is empty.

## 9. Open questions

1. Who uses the system besides the owner of one bot? Is a multi-chat setup intended?
2. Which phone, Android version and Python version is it meant for? Nothing in the repository says.
3. What is the provenance of `yolov8n_float16.tflite`? Answered by the file itself: the `metadata.json` appended to it reads "Ultralytics YOLOv8n model trained on coco.yaml", Ultralytics 8.4.33, exported 2026-04-05, with 80 names where 0 is "person" and 58 is "potted plant". As far as the metadata shows, it is stock COCO YOLOv8n. The same metadata names the AGPL-3.0 licence, while `LICENSE` is MIT. Still open: is counting COCO "potted plant" the intended "leaf count"?
4. How were the HSV bounds [35,40,40]..[85,255,255] chosen, and the 50 px contour filter?
5. Is publishing the photos on an anonymous public host (catbox.moe) acceptable? Should they ever be deleted?
6. Is `vision_mqtt.py` still maintained? Should it share `config.py` and `TelegramBot` with the plant monitor?
7. Should the README point to `main` instead of `plant-growth-metrics-termux`? Partly answered: `git ls-remote --heads origin` lists only `main`, so the clone command in `README.md:66` fails as written ("Remote branch plant-growth-metrics-termux not found in upstream origin", exit 128, reproduced against a local clone), and the plant monitor code is on `main`.
8. What will provide temperature and humidity: which MQTT topic and which sensor?
9. Timestamps are device local time with no zone. Is that intended for the sheet?
10. Which paho-mqtt version is targeted? `mqtt.Client()` with no arguments (`vision_mqtt.py:103`) raises `TypeError` in paho-mqtt 2.0.0, where `callback_api_version` is required; 2.1.0 and 2.2.0 accept it with a `DeprecationWarning`. Read from the published wheels, not run; 1.x not checked.
