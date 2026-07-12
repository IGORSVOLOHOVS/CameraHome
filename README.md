# CameraHome 📸 & Plant Growth Monitor 🌱
**Autonomous Edge Vision & Metrics Surveillance for Android (Termux)**

An AI-powered edge computing ecosystem designed to run entirely on an Android phone via Termux. It includes two primary modules:
1. **Plant Growth Monitor 🌱**: Tracks plant growth, detects leaves/pots using YOLOv8, outlines green mass contours (HSV), and automatically logs metrics to Google Sheets alongside instant Telegram reports.
2. **Security Surveillance 📸**: Detects people in real-time and sends alerts with annotated photos to Telegram and Home Assistant (via MQTT).

---

## ✨ Features (Plant Growth Monitor)
- 🚀 **Edge AI**: Runs YOLOv8n (TFLite) local inference on mobile hardware.
- 📐 **Precision Growth CV**: Outlines green sprout/leaf contours using HSV color space and calculates green mass coverage percentage.
- 💾 **Local Resilience**: Caches captured data in a local SQLite database if internet connection is down.
- 📊 **Google Sheets Sync**: Automatically syncs database records to your Google Sheet with embedded clickable previews (renders images directly inside cells!).
- ☁️ **No-Quota Hosting**: Automatically uploads annotated snapshots to a free, anonymous hosting service (Catbox) to bypass Google Drive's zero-quota service account limits.
- 💬 **Telegram Bot Commands**:
  - `/photo` - Takes a photo and analyzes it immediately.
  - `/status` - Reports system mode, cache status, and last measured metrics.
  - `/interval <hours>` - Updates measurement frequency dynamically (e.g. `/interval 2` or `/interval 0.5`).
  - `/sheets` - Returns a direct link to the active Google Sheet.
  - `/simulate` - Toggles 1-minute simulation intervals for fast testing.
  - `/update` - Performs a `git pull` from GitHub and hot-restarts the python script seamlessly.

---

## 🛠️ System Requirements

### 1. Android Applications
- **[Termux](https://f-droid.org/en/packages/com.termux/)** (F-Droid version only)
- **[Termux:API](https://f-droid.org/en/packages/com.termux.api/)** (Required for accessing the phone's hardware camera. Make sure to grant Camera permissions to Termux:API in Android settings).

### 2. Google Cloud Platform Setup (for Google Sheets)
1. Go to [Google Cloud Console](https://console.cloud.google.com/).
2. Enable both **Google Sheets API** and **Google Drive API** in your project.
3. Create a **Service Account** and download its key in **JSON** format.
4. Rename this file to `credentials.json` and place it in the root folder of the project.
5. Create a Google Sheet, and share it with the service account's email (found in `credentials.json`) with **Editor** permissions.

---

## 🚀 Installation & Setup in Termux

Open Termux on your phone and run the following setup commands:

### Step 1: Install System Packages
Enable the X11 repository (needed for OpenCV) and install dependencies:
```bash
pkg update && pkg upgrade -y
pkg install x11-repo -y
pkg update
pkg install termux-api python python-cryptography python-numpy python-pillow opencv-python git dbus -y
```

### Step 2: Install Python Libraries
Install Python packages that do not require C/C++ compilation:
```bash
pip install gspread google-auth requests tflite-runtime
```

### Step 3: Clone the Repository & Configure
Clone your repository and switch to the plant monitor branch:
```bash
git clone -b plant-growth-metrics-termux https://github.com/IGORSVOLOHOVS/CameraHome.git
cd CameraHome
```

Copy the configuration template:
```bash
cp .env.example .env
nano .env
```
Fill in the configuration details:
- `TELEGRAM_TOKEN` - Your bot token from @BotFather.
- `SPREADSHEET_ID` - The ID of your Google Sheet (extracted from its URL).
- `TELEGRAM_CHAT_ID` - Can be left as `your_telegram_chat_id` (the bot will auto-discover it on the first message you send to it).

Move your downloaded `credentials.json` (Google Service Account key) into this directory.

---

## 🎮 Running the Application
Start the monitoring service:
```bash
python main.py
```
Open Telegram, search for your bot, and send `/start` or `/help` to see available commands!

---

## ⚙️ Telegram Autocomplete Commands Setup
To register command auto-complete suggestions in Telegram:
1. Write to **[@BotFather](https://t.me/BotFather)**.
2. Send `/setcommands` and choose your bot.
3. Paste the following list:
```text
photo - Сделать снимок и запустить анализ
status - Показать статус системы и метрики
interval - Изменить частоту замеров в часах
sheets - Получить ссылку на Google Таблицу
simulate - Вкл/выкл режим симуляции (1 мин)
update - Обновить код с GitHub и перезапустить
```
