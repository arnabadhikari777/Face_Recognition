# মডেল অটো-ডাউনলোড গাইড

আপনাকে কোনো `.onnx` ফাইল নিজে ডাউনলোড বা কপি করতে হবে না। Render build-এর সময় সব মডেল নিজে নেমে আসে।

## কীভাবে কাজ করে
1. `models/model_manifest.json` — সব মডেলের নাম, URL, SHA-256, সাইজ, ইনপুট/আউটপুট, লাইসেন্স এখানে আছে (একমাত্র উৎস)।
2. `scripts/download_models.sh` → `scripts/download_models.py` (শুধু Python standard library):
   * InsightFace zip (`buffalo_s.zip` ≈ 127.6 MB, `buffalo_l.zip` ≈ 288.6 MB) নামায়, SHA-256 + সাইজ মেলায়, দরকারি `.onnx` বের করে, zip মুছে ফেলে।
   * YuNet (≈ 0.23 MB) সরাসরি নামায়।
   * প্রতিটি `.onnx`-এর SHA-256, সাইজ এবং ONNX হেডার যাচাই হয় (Git LFS pointer / HTML error পেজ ধরা পড়ে)।
   * যাচাই করা ফাইল আবার নামানো হয় না; নষ্ট ফাইল মুছে নতুন করে নামানো হয়।
   * নেটওয়ার্ক ফেল করলে `DOWNLOAD_RETRIES` (ডিফল্ট ৪) বার আবার চেষ্টা করে।
   * কোনো required মডেল প্রস্তুত না হলে exit code 1 → Render build **ফেল** করে (চুপচাপ বাদ পড়ে না)।
3. Non-commercial লাইসেন্সের মডেল নামাতে `ACCEPT_NONCOMMERCIAL_MODEL_LICENSES=1` লাগে (দেখুন `MODEL_LICENSES.md`)।

## ফলাফল (static/models/)
| ফাইল | সাইজ |
|---|---|
| `det_500m.onnx` (SCRFD) | 2,524,817 B |
| `face_detection_yunet_2023mar.onnx` | 232,589 B |
| `w600k_mbf.onnx` | 13,616,099 B |
| `w600k_r50.onnx` | 174,383,860 B |
মোট ≈ 190.8 MB। ব্রাউজার প্রথমবার নামায়, SHA-256 মিলিয়ে Cache API-তে রাখে; পরের বার আর নামে না।

## Environment variable (শুধু build-এর সময়)
`ACCEPT_NONCOMMERCIAL_MODEL_LICENSES`, `DOWNLOAD_RETRIES`, `MODELS_DEST` (ডিফল্ট `static/models`), `MODEL_SEED_DIR` (অফলাইনে আগে থেকে নামানো ফাইলের ফোল্ডার)।

## নতুন মডেল যোগ/বদল
`model_manifest.json`-এ নতুন এন্ট্রি (সঠিক SHA-256 সহ), `static/js/engine/recognizers.js`-এ ওই মডেলের নিজস্ব preprocessing, `models/calibration.json`-এ থ্রেশহোল্ড। নতুন recognition মডেলের জন্য পুরনো মানুষদের টেমপ্লেট নতুন স্পেসে বদলানো যায় না — `DATABASE_MIGRATION_BN.md` দেখুন।

## যাচাই (আমি যা করেছি / করিনি)
* করেছি: লাইসেন্স গেট, চেকসাম, "আগে যাচাই হয়েছে → skip", নষ্ট ফাইল মেরামত — সব আমার স্যান্ডবক্সে চালিয়ে দেখেছি (InsightFace zip আসল ডাউনলোড; YuNet লোকাল সিড থেকে)।
* করিনি: আমার স্যান্ডবক্সে YuNet-এর আসল URL (`media.githubusercontent.com` রিডাইরেক্ট) ব্লক ছিল, তাই Render-এ সেই এক ফাইলের নেটওয়ার্ক ডাউনলোড আমি নিজে চালাইনি। SHA-256 মেলানো আছে বলে ভুল ফাইল ঢুকবে না — না নামলে build ফেল করবে।
