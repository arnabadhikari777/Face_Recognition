# সমস্যা ও সমাধান

| লক্ষণ | কারণ / সমাধান |
|---|---|
| Render build ফেল: "non-commercial … ACCEPT_NONCOMMERCIAL_MODEL_LICENSES" | env var `ACCEPT_NONCOMMERCIAL_MODEL_LICENSES=1` দিন (শুধু যদি ব্যবহার non-commercial হয়; `MODEL_LICENSES.md`)। |
| build ফেল: "download failed" / "SHA-256 mismatch" | নেটওয়ার্ক বা আপস্ট্রিম ফাইল বদলেছে। Manual Deploy → Clear build cache & deploy। SHA বদলালে `model_manifest.json` সঠিক হ্যাশ দিয়ে ঠিক করুন — হ্যাশ না মিলিয়ে ফাইল গ্রহণ করা হয় না। |
| পেজে "model files are missing on the server" | build-এ ডাউনলোড হয়নি/`static/models/` খালি। Build log-এ `[download_models] All 4 models are ready` খুঁজুন। |
| "failed its integrity check" | ডাউনলোড ভেঙেছে। পেজ রিলোড করুন; বারবার হলে সার্ভারের ফাইল নষ্ট → রি-ডিপ্লয়। |
| লোডিংয়ে ক্র্যাশ/ধীর (বিশেষ করে ফোনে) | Accuracy মোডে IResNet-50 (≈174 MB) লাগে। Balanced বা Compatibility বেছে নিন। |
| "Cannot reach the server" | Render Free ঘুম থেকে উঠতে ~৫০ সেকেন্ড লাগে; এক মিনিট পর আবার চেষ্টা। |
| 503 database_unavailable | `MONGODB_URI` ভুল, অথবা Atlas Network Access-এ `0.0.0.0/0` নেই, অথবা ইউজার/পাসওয়ার্ড ভুল। `/api/health` দেখুন। |
| 503 / `decryption failed` | `TEMPLATE_ENCRYPTION_KEY` বদলে গেছে/মিসিং। আগের key ফেরত দিন। key হারালে এনক্রিপ্টেড টেমপ্লেট পড়া যায় না। |
| 409 model_mismatch | ব্রাউজারে পুরনো পেজ। হার্ড রিলোড। |
| 422 insufficient_samples / inconsistent_samples / low_quality | ভালো আলো, স্থির মাথা, একজনই ফ্রেমে; আবার রেজিস্টার। |
| 503 busy | আরেকটি রেজিস্ট্রেশন চলছে; কয়েক সেকেন্ড পরে। |
| ক্যামেরা চলে না | HTTPS লাগে (Render-এ আছে); ব্রাউজারে ক্যামেরা অনুমতি দিন। |
| বারবার UNCERTAIN | থ্রেশহোল্ড অ-ক্যালিব্রেটেড — `MODEL_BENCHMARK_BN.md` |
| `/api/health` → `"encryption": "disabled"` | `TEMPLATE_ENCRYPTION_KEY` সেট নেই; ঐচ্ছিক কিন্তু সুপারিশকৃত। |
