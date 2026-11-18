# .env সেটআপ গাইড (বাংলা)

এই ফাইলটি শুধু নির্দেশিকা। আসল সিক্রেট কখনো GitHub-এ কমিট করবেন না।

## ১. লোকাল `.env` ফাইল

প্রজেক্ট রুটে `.env.example` কপি করে `.env` নামে সেভ করুন:

```bash
cp .env.example .env
```

তারপর `.env` এডিট করুন।

### MongoDB Atlas (`MONGODB_URI`)

1. [MongoDB Atlas](https://cloud.mongodb.com/) এ লগইন করুন।
2. Database → Connect → Drivers → connection string কপি করুন।
3. স্ট্রিংয়ের এই অংশগুলো বদলান:
   - `<DB_USER>` → আপনার database username
   - `<DB_PASSWORD>` → পাসওয়ার্ড
   - `<CLUSTER_HOST>` → আপনার cluster hostname (যেমন `cluster0.xxxxx.mongodb.net`)

**পাসওয়ার্ডে স্পেশাল ক্যারেক্টার থাকলে URL-encode করুন:**

| ক্যারেক্টার | এনকোড |
|-------------|--------|
| `@`         | `%40`  |
| `#`         | `%23`  |
| `/`         | `%2F`  |
| `:`         | `%3A`  |
| `?`         | `%3F`  |
| `%`         | `%25`  |
| ` ` (স্পেস) | `%20`  |

উদাহরণ (কাজ করবে না — শুধু ফরম্যাট):

```
MONGODB_URI=mongodb+srv://myuser:p%40ssw0rd@cluster0.abc12.mongodb.net/?retryWrites=true&w=majority
```

অন্যান্য:

```
MONGODB_DB_NAME=face_recognition_db
MONGODB_COLLECTION=identities
```

এগুলো সাধারণত অপরিবর্তিত রাখুন।

### টেমপ্লেট এনক্রিপশন (`TEMPLATE_ENCRYPTION_KEY`)

একবার জেনারেট করুন এবং নিরাপদে রাখুন:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

আউটপুট যেমন: `QQObYN5QEnpW7NHJAVAU_oNktiVN92f5Mdz6JclfeXg=`

`.env`-এ বসান:

```
TEMPLATE_ENCRYPTION_KEY=QQObYN5QEnpW7NHJAVAU_oNktiVN92f5Mdz6JclfeXg=
```

**গুরুত্বপূর্ণ:** ইতিমধ্যে এনক্রিপ্টেড রেকর্ড থাকলে **একই কী** ব্যবহার করুন। কী হারালে পুরনো টেমপ্লেট পড়া যাবে না। অ্যাপ এনক্রিপ্টেড ও আনএনক্রিপ্টেড দুই ধরনের টেমপ্লেট সাপোর্ট করে।

### অন্যান্য ভ্যারিয়েবল

| নাম | কী করবেন |
|-----|----------|
| `MANIFEST_FILE` / `CALIBRATION_FILE` | খালি রাখুন (ডিফল্ট ফাইল ব্যবহার হবে) |
| `ENROLL_REQUIRE_ALL` | `false` রাখুন |
| `RETENTION_DAYS` | `0` = চিরকাল রাখা |
| `MAX_RECORDS` | `1500` (Atlas M0 সুরক্ষা) |
| `TRUST_PROXY` | Render-এ `true` |
| `ALLOWED_ORIGINS` | সাধারণত খালি (same-origin) |
| `MAX_CONTENT_LENGTH` | `524288` |
| `RATE_*` | ডিফল্ট রাখুন |
| `ACCEPT_NONCOMMERCIAL_MODEL_LICENSES` | build-এর জন্য `1` |

### যে ভ্যারিয়েবলগুলো **ব্যবহার করবেন না**

পুরনো/অসমর্থিত:

- `MONGODB_DB` (সঠিক নাম: `MONGODB_DB_NAME`)
- `CORS_ORIGINS` (সঠিক নাম: `ALLOWED_ORIGINS`)
- `FLASK_DEBUG`
- `SECRET_KEY`
- `DUPLICATE_THRESHOLD` / `RECOGNITION_THRESHOLD` / `UNCERTAIN_THRESHOLD` (এগুলো `models/calibration.json` থেকে আসে)

## ২. Render Dashboard-এ Environment

লোকাল `.env` Render-এ অটো লোড হয় না।

Render → আপনার Web Service → **Environment** ট্যাবে উপরের ভ্যারিয়েবলগুলো আলাদা আলাদা করে যোগ করুন।

বিশেষ করে বাধ্যতামূলক:

- `MONGODB_URI`
- `ACCEPT_NONCOMMERCIAL_MODEL_LICENSES=1`
- `TRUST_PROXY=true`
- (সুপারিশ) `TEMPLATE_ENCRYPTION_KEY`

## ৩. যাচাই

লোকাল:

```bash
flask --app app run
# অন্য টার্মিনালে:
curl -s http://127.0.0.1:5000/api/health
curl -s http://127.0.0.1:5000/api/config | head
```

Render:

```
https://<your-app>.onrender.com/api/health
https://<your-app>.onrender.com/api/config
```
