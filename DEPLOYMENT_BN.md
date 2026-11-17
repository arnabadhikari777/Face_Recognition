# ডিপ্লয়মেন্ট গাইড (বাংলা): MongoDB Atlas + GitHub + Render

> নিচের ধাপগুলো আমি লাইভ Atlas/Render অ্যাকাউন্টে চালিয়ে দেখতে পারিনি। ড্যাশবোর্ডের বোতামের নাম বদলে গেলে কাছাকাছি নামেরটা খুঁজে নিন।

## ১. MongoDB Atlas (ডাটাবেস)
1. mongodb.com/atlas এ অ্যাকাউন্ট খুলুন → **Create** → **M0 Free** → কাছের রিজিয়ন → ক্লাস্টার তৈরি হওয়া পর্যন্ত অপেক্ষা করুন।
2. **Database Access** → **Add New Database User** → Password পদ্ধতি। ইউজারনেম `face_app`, পাসওয়ার্ড অটো-জেনারেট করে সেভ করে রাখুন। রোল: **readWrite** শুধু `face_recognition_db` ডাটাবেসে।
3. **Network Access** → **Add IP Address** → **Allow access from anywhere** (`0.0.0.0/0`)। Render Free-এর স্থির IP নেই, তাই এটা লাগবে।
4. **Database** → **Connect** → **Drivers** → `mongodb+srv://...` স্ট্রিং কপি করুন। `<db_user>` ও `<db_password>` বদলান। পাসওয়ার্ডে বিশেষ চিহ্ন থাকলে এনকোড করুন (`@` → `%40`, `#` → `%23`)।

## ২. GitHub-এ আপলোড
1. github.com-এ একটা খালি রিপোজিটরি বানান (README/license/.gitignore যোগ করবেন না)।
2. প্রজেক্ট ফোল্ডারে:
```
git init
git add .
git status        # .env এবং কোনো .onnx ফাইল তালিকায় থাকা চলবে না
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<ইউজার>/<রিপো>.git
git push -u origin main
```
`.env` বা কানেকশন স্ট্রিং কখনো GitHub-এ দেবেন না।

## ৩. Render-এ ডিপ্লয়
**পদ্ধতি ক (সহজ):** Render → **New +** → **Blueprint** → রিপো বাছুন। `render.yaml` পড়ে সব সেট করবে, শুধু `MONGODB_URI` চাইলে Atlas-এর স্ট্রিংটা বসান।

**পদ্ধতি খ (হাতে):** **New +** → **Web Service** → রিপো বাছুন, তারপর:
| ঘর | মান |
|---|---|
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt && sh scripts/download_models.sh` |
| Start Command | `gunicorn app:app --workers 1 --threads 4 --timeout 60 --bind 0.0.0.0:$PORT` |
| Instance Type | Free |
| Health Check Path | `/api/health` |

Environment Variables:
| Key | Value |
|---|---|
| `MONGODB_URI` | Atlas-এর স্ট্রিং |
| `TRUST_PROXY` | `true` |
| `PYTHON_VERSION` | `3.12.3` |

**Create Web Service** চাপুন। প্রথম বিল্ডে মডেল ফাইল (প্রায় ৩৭ MB) GitHub থেকে নামবে, তাই একটু সময় লাগবে।

## ৪. ডিপ্লয়ের পর পরীক্ষা
1. `https://<আপনার-সার্ভিস>.onrender.com/api/health` → `{"database":"ok","status":"ok"}` আসা উচিত (ঘুমিয়ে থাকলে ১ মিনিট পর্যন্ত লাগতে পারে)।
2. `.../api/config` → `opencv-zoo-sface`, `2021dec`, `128`।
3. হোম পেজে **Register Face** → সম্মতি টিক → নাম → **Start Camera** → ৩টি ছবি (সামনে, বাঁয়ে, ডানে)।
4. একই মুখ আবার রেজিস্টার করলে "Face already registered" আসবে।
5. **Recognize Face** দিয়ে মিলিয়ে দেখুন।
6. Atlas → **Browse Collections** এ দেখুন ডকুমেন্টে নাম, ৩টি embedding, `created_at` আছে এবং কোনো ছবি নেই।

## ৫. সমস্যা হলে
| সমস্যা | কারণ / সমাধান |
|---|---|
| Build-এ `download_models.sh` ফেইল | GitHub থেকে মডেল নামেনি। Render-এর বিল্ড লগ দেখুন, আবার Deploy দিন। |
| `/api/health` এ `database: unavailable` | `MONGODB_URI` ভুল, পাসওয়ার্ড এনকোড করা হয়নি, অথবা `0.0.0.0/0` যোগ হয়নি। |
| "face model files are missing" | বিল্ড লগে দেখুন `.onnx` ফাইল নেমেছে কি না। |
| ক্যামেরা চালু হয় না | HTTPS লাগবে (Render URL ঠিক আছে), ব্রাউজারে ক্যামেরা পারমিশন দিন। |
| প্রথম রিকোয়েস্ট খুব ধীর | Render Free ১৫ মিনিট নিষ্ক্রিয় থাকলে ঘুমায়; ১ মিনিট অপেক্ষা করুন। |
| ৪২৯ এরর | প্রতি IP-র রেট লিমিট; একটু পর চেষ্টা করুন। |

## ৬. গুরুত্বপূর্ণ সতর্কতা
- Embedding বায়োমেট্রিক ডেটা। জনসমক্ষে চালু করার আগে আপনার দেশের গোপনীয়তা আইন দেখে নিন।
- থ্রেশহোল্ড (`0.363`) আপনার ক্যামেরা/ব্যবহারকারীর জন্য যাচাই করা নয়। আসল ডেটায় পরীক্ষা করে ঠিক করুন।
- ছবির লাইভনেস ডিটেকশন নেই: কারও ছবি দেখিয়ে ধোঁকা দেওয়া সম্ভব।
