# ডেপ্লয়মেন্ট গাইড — ফেস রিকগনিশন ওয়েব অ্যাপ

এই নথিতে MongoDB Atlas, মডেল ফাইল, GitHub এবং Render Free-তে ডেপ্লয় করার পদ্ধতি তুলে ধরা হয়েছে।

---

## A. MongoDB Atlas Free ক্লাস্টার তৈরি

1. https://www.mongodb.com/cloud/atlas-এ যান এবং সাইন আপ / লগ ইন করুন।
2. একটি **Free (M0)** ক্লাস্টার তৈরি করুন। ব্যবহারকারীদের কাছাকাছি কোনো অঞ্চল বেছে নিন।
3. ক্লাস্টারের অবস্থা **Available** হওয়া পর্যন্ত অপেক্ষা করুন।

---

## B. ডেটাবেস ব্যবহারকারী তৈরি

1. Atlas → **Database Access** → **Add New Database User**-এ যান।
2. Authentication: Password নির্বাচন করুন।
3. Username ও শক্তিশালী password দিন (দুটিই সংরক্ষণ করুন)।
4. Privileges: **Read and write to any database** নির্বাচন করুন (অথবা `face_recognition_db`-তে সীমাবদ্ধ রাখুন)।
5. ব্যবহারকারী তৈরি করুন।

---

## C. নিরাপদ নেটওয়ার্ক অ্যাক্সেস কনফিগার

1. Atlas → **Network Access** → **Add IP Address**-এ যান।
2. Render-এর ডায়নামিক IP-এর জন্য সাধারণত **Allow Access from Anywhere** (`0.0.0.0/0`) চালু করতে হয়।
   - বিনামূল্যের সার্ভারলেস/PaaS হোস্টে এটি প্রচলিত। পরে স্ট্যাটিক outbound IP যোগ করলে অ্যাক্সেস সীমিত করুন।
3. নিশ্চিত করুন।

**দ্রষ্টব্য:** ক্লাস্টারকে authentication ছাড়া রাখবেন না। ডেটাবেস ব্যবহারকারীর জন্য সব সময় শক্তিশালী password ব্যবহার করুন।

---

## D. MongoDB সংযোগ স্ট্রিং সংগ্রহ

1. Atlas → **Database** → **Connect** → **Drivers**-এ যান।
2. SRV string কপি করুন, যেমন:
   ```
   mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```

mongodb+srv://face_admin:zwmDawYyq6H7TGHf@arnab.r8slxp9.mongodb.net/?retryWrites=true&w=majority
mongodb+srv://face_admin:zwmDawYyq6H7TGHf@arnab.r8slxp9.mongodb.net/face_recognition_db?retryWrites=true&w=majority

3. `<username>` এবং `<password>`-এর জায়গায় আপনার তথ্য বসান (password-এ বিশেষ অক্ষর থাকলে URL-encode করুন)।
4. ঐচ্ছিক: ডেটাবেসের নাম যোগ করুন: `...mongodb.net/face_recognition_db?retryWrites=true&w=majority`

---

## E. মডেল ফাইল (আবশ্যক)

ONNX Runtime Web দিয়ে **ব্রাউজারেই** মুখ শনাক্তকরণ ও embedding তৈরি হয়।

`static/models/`-এর মধ্যে অন্তত একটি **recognizer** ONNX ফাইল রাখুন:

| ফাইল (যেকোনো একটি নাম চলবে) | কাজ | সাধারণ আকার |
|---------------------------------|------|--------------|
| `w600k_mbf.onnx` / `mobilefacenet.onnx` | 512-d embedding (প্রস্তাবিত) | ~4–14 MB |
| `det_10g.onnx` / `ultraface-rfb-320.onnx` | ঐচ্ছিক detector | ~1–5 MB |

### প্রস্তাবিত উৎস (পুনর্বিতরণের আগে লাইসেন্স যাচাই করুন)

- InsightFace model zoo (buffalo_s / MobileFaceNet `w600k_mbf`):  
  https://github.com/deepinsight/insightface/tree/master/model_zoo  
  অফিসিয়াল প্যাকগুলো GitHub Releases / InsightFace wiki-তে থাকা Google Drive লিংকে প্রকাশিত হয়।
- EdgeFace-XXS / XS (Apache-সমর্থিত ছোট মডেল):\
  https://github.com/yakhyo/edgeface-onnx

**লাইসেন্স:** ডাউনলোড করা নির্দিষ্ট ফাইলটি পুনর্বিতরণ করা যাবে কি না নিশ্চিত করুন। লাইসেন্সে ফাইলটি প্যাকেজ করার অনুমতি না থাকলে মডেলগুলো পাবলিক রিপোজিটরির বাইরে রাখুন এবং ডেপ্লয়কারীদের জন্য ডাউনলোডের ধাপ লিখে দিন।

ডাউনলোডের পর:

```bash
mkdir -p static/models
# copy your .onnx files into static/models/
```

অ্যাপে ব্যবহৃত preprocessing (মডেলটি যে পদ্ধতিতে প্রশিক্ষিত, তার সঙ্গে মিলতে হবে):

- ইনপুট: 112×112 RGB মুখের crop
- Normalize: `(pixel - 127.5) / 127.5`
- আউটপুট: L2-normalized embedding (MobileFaceNet ArcFace-এর জন্য 512-d)

recognizer মডেল না থাকলে UI-তে ONNX ফাইল যোগ করার জন্য স্পষ্ট ত্রুটি বার্তা দেখানো হবে।

---

## F. প্রজেক্ট GitHub-এ আপলোড

1. GitHub-এ একটি নতুন **public** repository তৈরি করুন (লোকালেই README থাকলে README ছাড়া তৈরি করুন)।
2. লোকাল মেশিনে:

```bash
cd face-recognition
git init
git add .
git commit -m "Initial face recognition web app"
git branch -M main
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main
```

**কখনোই commit করবেন না** `.env`, প্রকৃত credentials বা প্রকৃত biometric data।

---

## G. GitHub-কে Render-এর সঙ্গে যুক্ত করুন

1. https://render.com-এ সাইন আপ করুন।
2. **New** → **Web Service** নির্বাচন করুন।
3. GitHub account যুক্ত করে repository নির্বাচন করুন।
4. কনফিগার করুন:
   - **Name:** যেমন `face-recognition`
   - **Runtime:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `Procfile` থাকলে ফাঁকা রাখুন; Render সেটি ব্যবহার করবে।
   - **Instance type:** Free

---

## H. Flask অ্যাপ্লিকেশন ডেপ্লয়

1. Render → **Environment**-এ নিচের ভেরিয়েবলগুলো যোগ করুন:

| Key | Value |
|-----|--------|
| `MONGODB_URI` | আপনার Atlas connection string |
| `MONGODB_DB` | `face_recognition_db` |
| `SECRET_KEY` | দীর্ঘ random string |
| `CORS_ORIGINS` | আপনার Render URL, অথবা পরীক্ষার জন্য `*` |
| `DUPLICATE_THRESHOLD` | `0.55` (ঐচ্ছিক) |
| `RECOGNITION_THRESHOLD` | `0.48` (ঐচ্ছিক) |

2. Deploy করুন। প্রথম build শেষ হওয়া পর্যন্ত অপেক্ষা করুন।
3. সার্ভিসের URL খুলুন।

**Free tier সম্পর্কে:**

- কিছুক্ষণ ব্যবহার না হলে সার্ভিস sleep করে → ঘুম থেকে ওঠার পর প্রথম request-এ cold start হয় (30–60+ seconds)।
- RAM সীমিত (~512 MB)। এই অ্যাপ সার্ভারে AI মডেল লোড করে **না**।
- স্থায়ী local disk নেই — সব মুখের রেকর্ড MongoDB Atlas-এ থাকে।

---

## I. ডেপ্লয় করা ওয়েবসাইট পরীক্ষা

1. `https://YOUR-SERVICE.onrender.com` খুলুন।
2. Health পরীক্ষা করুন: `https://YOUR-SERVICE.onrender.com/api/health`
   প্রত্যাশিত ফল: `"database": "connected"`।
3. মডেল ফাইল যোগ করার পর একটি মুখ নিবন্ধন করুন, তারপর সেটি শনাক্ত করুন।
4. একই মুখ আবার নিবন্ধনের চেষ্টা করুন — 409-ধরনের বার্তা আসা উচিত।

---

## J. Android Chrome-এ ক্যামেরার অনুমতি

1. **HTTPS** ব্যবহার করুন (Render HTTPS দেয়)।
2. তালা / site settings-এ ট্যাপ করে Camera অনুমতি দিন।
3. আগে অনুমতি প্রত্যাখ্যান করা হয়ে থাকলে Chrome settings থেকে site permission reset করুন।
4. Selfie-এর জন্য সামনের ক্যামেরা ব্যবহার করুন; পেছনের ক্যামেরার জন্য **Switch Camera** চাপুন।

---

## K. সমস্যা সমাধান

| সমস্যা | যা পরীক্ষা করবেন |
|-------|----------------|
| Health-এ database unreachable দেখায় | `MONGODB_URI`, Atlas ব্যবহারকারীর password, Network Access-এ `0.0.0.0/0` |
| মডেল লোড হয় না | `static/models/`-এর ফাইল, সঠিক `.onnx` নাম, browser console |
| CORS ত্রুটি | `CORS_ORIGINS`-এ frontend origin আছে কি না |
| 429 Too many requests | Rate limit settings; 60 seconds অপেক্ষা করুন |
| Cold start | Free tier-এর sleep; অপেক্ষা করে আবার চেষ্টা করুন |
| মোবাইলে ক্যামেরা কাজ করে না | HTTPS আবশ্যক; অনুমতি দেওয়া আছে কি না; অন্য কোনো app ক্যামেরা ব্যবহার করছে কি না |

---

## L. মালিকের অনুমোদনে ডেটা মুছে ফেলা

নকশা অনুযায়ী **সবার জন্য উন্মুক্ত Delete button নেই**।

কোনো নিবন্ধন মুছতে:

1. MongoDB Atlas-এ লগ ইন করুন → Browse Collections → `face_recognition_db` → `registered_faces`।
2. `name` বা `_id` দিয়ে নথিটি খুঁজুন।
3. নথিটি মুছুন।

অথবা একই URI ব্যবহার করে নিরাপদ shell থেকে:

```python
from pymongo import MongoClient
from bson import ObjectId
client = MongoClient("YOUR_URI")
col = client["face_recognition_db"]["registered_faces"]
col.delete_one({"_id": ObjectId("...")})
# or
col.delete_many({"name": "Person Name"})
```

Biometric data-এর জন্য আপনার স্থানীয় গোপনীয়তা বিধি অনুযায়ী data-retention policy তৈরি ও অনুসরণ করুন।

---

## লোকাল ডেভেলপমেন্ট

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env with a local or Atlas URI
python app.py
```

http://127.0.0.1:5000 খুলুন।

Production-এর কাছাকাছি পরিবেশে চালাতে:

```bash
gunicorn app:app --bind 0.0.0.0:5000
```

---

## Environment Variables-এর সারাংশ

| Variable | আবশ্যক | বিবরণ |
|----------|----------|-------------|
| `MONGODB_URI` | হ্যাঁ | Atlas connection string |
| `MONGODB_DB` | না | ডিফল্ট `face_recognition_db` |
| `SECRET_KEY` | হ্যাঁ (prod) | Flask secret |
| `CORS_ORIGINS` | না | `*` অথবা কমা দিয়ে আলাদা করা origins |
| `DUPLICATE_THRESHOLD` | না | Duplicate-এর cosine threshold (ডিফল্ট 0.55) |
| `RECOGNITION_THRESHOLD` | না | Match-এর cosine threshold (ডিফল্ট 0.48) |
| `UNCERTAIN_THRESHOLD` | না | এর নিচে হলে unknown (ডিফল্ট 0.40) |
| `RATE_LIMIT_REQUESTS` | না | ডিফল্ট 30 |
| `RATE_LIMIT_WINDOW_SECONDS` | না | ডিফল্ট 60 |
| `FLASK_DEBUG` | না | শুধু লোকাল debug-এর জন্য `1` |
| `PORT` | স্বয়ংক্রিয় | Render সেট করে |

---

## আর্কিটেকচার (সংক্ষেপে)

- **Browser:** Camera → ONNX Runtime Web (WASM / ঐচ্ছিক WebGPU) → face embedding।
- **Server (Render):** Flask + Gunicorn — যাচাই করে, MongoDB-তে embedding সংরক্ষণ/তুলনা করে।
- **Database:** MongoDB Atlas free cluster — শুধু name + embeddings + model metadata সংরক্ষণ করে (কোনো ছবি নয়)।
