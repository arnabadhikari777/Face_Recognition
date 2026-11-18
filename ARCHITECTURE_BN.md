# আর্কিটেকচার

## কে কী করে
| জায়গা | দায়িত্ব |
|---|---|
| ইউজারের ফোন (Web Worker, ONNX Runtime Web) | ক্যামেরা, ফেস ডিটেকশন, ল্যান্ডমার্ক, অ্যালাইনমেন্ট, কোয়ালিটি যাচাই, সব recognition মডেল চালানো, embedding তৈরি |
| Render (Flask) | ভ্যালিডেশন, টেমপ্লেট তৈরি/যাচাই, ম্যাচিং, র‍্যাঙ্কিং, সিদ্ধান্ত, কনফিগ ও মডেল ফাইল সার্ভ (কোনো নিউরাল নেটওয়ার্ক চলে না) |
| MongoDB Atlas | টেমপ্লেট, master টেমপ্লেট, মডেল ভার্সন, মেটাডেটা |
কাঁচা ছবি/ভিডিও কখনো সার্ভারে যায় না; শুধু সংখ্যার embedding + কোয়ালিটি সংখ্যা যায়।

## পাইপলাইন (static/js/engine/)
`detectors.js` (SCRFD 500M প্রাইমারি, YuNet ফলব্যাক/প্রোফাইল অনুযায়ী) → `preprocess.js` (৫-পয়েন্ট similarity align → 112×112) → `quality.js` → `recognizers.js` (প্রতিটি মডেলের নিজস্ব preprocessing: RGB, (x−127.5)/127.5, NCHW) → `engine-core.js` (সব জোড়া লাগায়) → `worker.js`/`engine-client.js` (থ্রেড, cancel, মেমোরি রিলিজ) → `model-store.js` (ডাউনলোড + SHA-256 + Cache API)।
Adaptive detection: প্রাইমারি ডিটেক্টর কিছু না পেলে বা স্কোর কম হলে তবেই ফলব্যাক চলে। Accuracy মোডে ক্যাপচারের সময় দ্বিতীয় ডিটেক্টরের সাথে ল্যান্ডমার্ক মিলিয়ে "landmark agreement" কোয়ালিটি মান বের হয়।

## প্রোফাইল (manifest-এ)
* **Accuracy (ডিফল্ট):** SCRFD+YuNet, MobileFaceNet + IResNet-50 দুটোই সবসময় চলে।
* **Balanced:** MobileFaceNet প্রাইমারি; IResNet-50 শুধু সার্ভার চাইলে (borderline/ambiguous) লোড হয়, শেষে মেমোরি থেকে ছাড়া হয়।
* **Compatibility:** YuNet + MobileFaceNet।
ডিভাইস না পারলে চুপচাপ মডেল কমানো হয় না — পরিষ্কার বার্তা দেখায় ও অন্য মোড বাছতে বলে।

## স্টোরেজ (স্কিমা v2, collection `identities`)
প্রতি মানুষের একটি ডকুমেন্ট: `models.<model_id>` = {version, dim, **master**, **prototypes** (front/left/right/up/down), template_count}; `templates[]` = প্রতিটি আলাদা টেমপ্লেট (template_id, model_id, model_version, dim, **এনক্রিপ্টেড embedding**, pose, quality, captured_at, template_version)। ভিন্ন মডেলের vector কখনো মেশানো/গড় করা হয় না; ভিন্ন মডেল-ভার্সনের টেমপ্লেট ম্যাচিংয়ে লোডই হয় না। `meta` (revision, ক্যাশ ইনভ্যালিডেশন) ও `locks` (TTL লক) collection আছে।
এনক্রিপশন: `TEMPLATE_ENCRYPTION_KEY` (Fernet) দিলে `f1:` প্রিফিক্সে এনক্রিপ্টেড; না দিলে `p1:` (এনক্রিপ্ট ছাড়া) এবং `/api/health` সেটা জানায়।

## রেজিস্ট্রেশন
পোজ প্ল্যান: front×3, left×2, right×2, up×1 (ঐচ্ছিক), down×1 (ঐচ্ছিক) = ৬–৯টি। প্রতিটি ধাপে বার্স্ট ক্যাপচার → কোয়ালিটি গেট পাস করা ফ্রেমের মধ্য থেকে সেরা বাছাই। ক্লায়েন্ট ও সার্ভার দুই জায়গাতেই: ডুপ্লিকেট (একই পোজে cos ≥ 0.97) বাদ, MAD-ভিত্তিক outlier বাদ, প্রতিটি মডেলে পারস্পরিক consistency যাচাই, কমপক্ষে ৬টি টেমপ্লেট না হলে প্রত্যাখ্যান। Master = quality-weighted গড় + L2 normalize। একই ব্যক্তির ডুপ্লিকেট রেজিস্ট্রেশন আগের মতোই ব্লক (409); একই নাম + নতুন মডেল হলে শুধু নতুন মডেলের টেমপ্লেট যোগ হয় ("upgraded")। সমান্তরাল রেজিস্ট্রেশন: MongoDB TTL লক + ইন-প্রসেস লক।

## রিকগনিশন অ্যালগরিদম
প্রতি ফ্রেম f, প্রতি মডেল m, প্রতি ব্যক্তি i:
`frame_score = 0.4·cos(q, master) + 0.4·mean(top-2 cos(q, টেমপ্লেট)) + 0.2·cos(q, pose prototype)`
ফ্রেমগুলো quality-weighted ও নিচের দিকের outlier বাদ দিয়ে (MAD) একত্র হয় → মডেলের স্কোর (cosine, সম্ভাব্যতা নয়)। প্রতিটি মডেলের **নিজস্ব** ক্যালিব্রেশনে `z = (score − threshold)/scale`; মডেলগুলো score-level fusion-এ `fused_z` (ওজনযুক্ত গড়)। ফলাফল:
* **MATCH:** fused_z ≥ 0, ≥ ৫০% ফ্রেম সমর্থন করে, মডেলগুলো একমত, প্রার্থীদের মধ্যে margin ≥ ১.০।
* **UNKNOWN:** fused_z < −১.০।
* **UNCERTAIN:** বাকি সব (কারণ: model_disagreement / ambiguous / low_support / borderline)।
* **need_secondary:** শুধু প্রাইমারি মডেল এসেছে, কিন্তু সিদ্ধান্ত borderline → ক্লায়েন্ট দ্বিতীয় মডেল লোড করে আবার পাঠায় (`final=true`)।
* **retry:** ফ্রেমের কোয়ালিটি কম → ভুল পরিচয় না দিয়ে আবার চেষ্টা করতে বলে।
কোনো "৯৯%" জাতীয় confidence বানানো হয় না। থ্রেশহোল্ড এখন **অ-ক্যালিব্রেটেড ডিফল্ট** (`models/calibration.json`, `calibrated:false`) — `MODEL_BENCHMARK_BN.md` অনুযায়ী নিজের ডেটায় ক্যালিব্রেট করুন।

## API
`GET /api/config` (মডেল স্পেক, প্রোফাইল, কোয়ালিটি/রেজিস্ট্রেশন কনফিগ; কোনো থ্রেশহোল্ড বা secret নেই), `GET /api/health`, `POST /api/register-face`, `POST /api/recognize-face`। কোনো রেসপন্সে embedding ফেরত যায় না।
