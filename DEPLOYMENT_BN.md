# ডিপ্লয়মেন্ট গাইড (GitHub → MongoDB Atlas → Render)

## ০. আগে জেনে নিন
* AI মডেল **ফোনে** চলে; Render শুধু API + স্ট্যাটিক ফাইল (মডেলসহ ≈ 191 MB) সার্ভ করে।
* InsightFace মডেলের লাইসেন্স non-commercial — `MODEL_LICENSES.md` পড়ুন।
* থ্রেশহোল্ড অ-ক্যালিব্রেটেড (`MODEL_BENCHMARK_BN.md`)। আসল ব্যবহারের আগে ক্যালিব্রেট করুন।

## ১. GitHub
1. নতুন প্রাইভেট রিপোজিটরি বানান।
2. ZIP খুলে সব ফাইল আপলোড/পুশ করুন। `.env` ও `static/models/*.onnx` আপলোড করবেন না (`.gitignore` ঠিক করা আছে; মডেল build-এ নামে)।

## ২. MongoDB Atlas
1. Free (M0) cluster বানান → Database Access: ইউজার + পাসওয়ার্ড → Network Access: `0.0.0.0/0` (Render-এর IP বদলায়)।
2. Connect → Drivers → `mongodb+srv://...` স্ট্রিং কপি করুন (এটাই `MONGODB_URI`)।
3. (সুপারিশকৃত) এনক্রিপশন key বানান: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` — নিরাপদে রাখুন, হারালে টেমপ্লেট পড়া যাবে না।

## ৩. Render
Dashboard → New + → Web Service → রিপোজিটরি সিলেক্ট (অথবা Blueprint, `render.yaml` ব্যবহার করে)।
* Runtime: Python 3
* **Build Command (হুবহু):**
  `pip install -r requirements.txt && sh scripts/download_models.sh`
* **Start Command (হুবহু):**
  `gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`
* Health Check Path: `/api/health`
* Environment variables:

| নাম | মান |
|---|---|
| `PYTHON_VERSION` | `3.12.3` |
| `ACCEPT_NONCOMMERCIAL_MODEL_LICENSES` | `1` (শুধু non-commercial ব্যবহারে) |
| `MONGODB_URI` | Atlas স্ট্রিং |
| `TEMPLATE_ENCRYPTION_KEY` | ধাপ ২.৩-এর key (ঐচ্ছিক) |
| `TRUST_PROXY` | `true` |
| ঐচ্ছিক | `MONGODB_DB_NAME`, `MONGODB_COLLECTION` (ডিফল্ট `identities`), `RETENTION_DAYS` (ডিফল্ট `0`), `MAX_RECORDS` (১৫০০), `ENROLL_REQUIRE_ALL`, `ALLOWED_ORIGINS`, `MAX_CONTENT_LENGTH`, `RATE_REGISTER_PER_HOUR`, `RATE_RECOGNIZE_PER_MINUTE`, `RATE_DEFAULT_PER_MINUTE`, `MANIFEST_FILE`, `CALIBRATION_FILE`, `DOWNLOAD_RETRIES` |

Deploy চাপুন। প্রথম build-এ ≈ 416 MB ডাউনলোড হয় (zip দুটি + YuNet), তারপর zip মুছে যায়; সময় নেটওয়ার্কের ওপর নির্ভর করে — আমি Render-এ নিজে চালিয়ে সময় মাপিনি।

## ৪. যাচাই
1. Build log-এ `[download_models] All 4 models are ready in .../static/models`।
2. `https://<আপনার-app>.onrender.com/api/health` → `{"status":"ok","database":"ok","encryption":"enabled","thresholds_calibrated":false}`।
3. `/api/config` খুলে ৪টি মডেল ও `default_profile: accuracy` দেখুন।
4. `/static/models/w600k_mbf.onnx` ডাউনলোড হয় কিনা দেখুন।
5. ফোনে পেজ খুলুন → Register Face → সম্মতি → Start Camera → মডেল লোড (প্রথমবার ≈ 191 MB; পরে ক্যাশ)।

## ৫. পুরনো ডেটা থাকলে
`DATABASE_MIGRATION_BN.md`।

## ৬. মডেল আপগ্রেড
`MODEL_DOWNLOAD_GUIDE_BN.md` (manifest বদলান → commit → Render রি-ডিপ্লয়)।

## ৭. লোকাল চালানো / টেস্ট
```
pip install -r requirements-dev.txt
ACCEPT_NONCOMMERCIAL_MODEL_LICENSES=1 sh scripts/download_models.sh
python -m pytest tests -q                       # সার্ভার টেস্ট (mongomock, সিন্থেটিক ভেক্টর)
npm install && npm test                         # JS ইউনিট টেস্ট; আসল-মডেল টেস্টের জন্য নিচের env দিন
python tests/reference/make_reference.py static/models <ছবির-ফোল্ডার> /tmp/ref.json
MODELS_DIR=static/models FIXTURES_DIR=<ছবির-ফোল্ডার> REF_JSON=/tmp/ref.json node --test tests/js/*.test.mjs
flask --app app run                              # লোকাল সার্ভার (.env থেকে কনফিগ)
```
