# বেঞ্চমার্ক ও থ্রেশহোল্ড ক্যালিব্রেশন

## সৎ অবস্থা
আসল, সম্মতিযুক্ত টেস্ট ডেটাসেট (বহু মানুষ, বিভিন্ন ফোন/আলো/পোজ/চশমা/সেশন) আমার কাছে ছিল না। তাই **কোনো "X% accuracy" দাবি করা হচ্ছে না।** থ্রেশহোল্ডগুলো `calibrated:false` (অ-ক্যালিব্রেটেড ইঞ্জিনিয়ারিং ডিফল্ট)। সিন্থেটিক টেস্ট শুধু কোডের যুক্তি প্রমাণ করে, রিয়েল-ওয়ার্ল্ড accuracy নয়।

## আমি যা মেপেছি (আসল মডেল, ৪–৬টি পাবলিক ফটো — শুধু sanity check, ভ্যালিডেশন নয়)
* আমার স্বাধীন Python রেফারেন্স (cv2 + onnxruntime) বনাম JS পাইপলাইন: alignment+tensor+ONNX+L2 (একই ল্যান্ডমার্কে) cosine ≥ **0.9994** (সহনসীমা ≥ 0.999); এন্ড-টু-এন্ড (ডিটেক্টর সহ) cosine 0.98–0.9998 (সহনসীমা ≥ 0.97; ~১ px ল্যান্ডমার্ক জিটারে embedding সংবেদনশীল), ল্যান্ডমার্ক ত্রুটি ≤ ০.৮৪% চোখের দূরত্ব (সহনসীমা ২%)।
* SCRFD বনাম YuNet ল্যান্ডমার্ক agreement: ০.৮৯–০.৯৫ (ল্যান্ডমার্ক ক্রমও মিলেছে)।
* একই ব্যক্তির ভিন্ন ছবি vs ভিন্ন ব্যক্তি (cosine): w600k_mbf ≈ 0.71 vs ≈ −0.03; w600k_r50 ≈ 0.78 vs ≈ −0.05 — **৩-৪টি ছবির উদাহরণ, পরিসংখ্যান নয়।**
* ডেস্কটপ CPU (onnxruntime-python) প্রতি ইনফারেন্স: mbf ≈ 20 ms, r50 ≈ 300 ms। ফোনে ধীর হবে; ব্রাউজারে মাপা হয়নি।
* বেঞ্চমার্ক টুলের নিজস্ব টেস্ট (FAR/FRR/EER/TAR@FAR/rank-1 গণনা যাচাই) পাস করেছে।

## আপনার ডেটায় বেঞ্চমার্ক চালানো
ফোল্ডার: `dataset/<person_id>/<condition>/<ছবি>.jpg` (condition = lighting, pose, glasses, phone_a ইত্যাদি, ঐচ্ছিক)। ন্যূনতম: ২০+ জন, ১০০+ genuine জোড়া, ১০ হাজার+ impostor জোড়া (FAR 1e-3 দাবি করতে ১০/FAR = ১০ হাজার impostor জোড়া)।
```
pip install -r requirements-dev.txt
sh scripts/download_models.sh          # (ACCEPT_NONCOMMERCIAL_MODEL_LICENSES=1 সহ)
python tools/benchmark.py --models static/models --dataset dataset --out benchmark_out
python tools/calibrate.py --results benchmark_out/results.json --far 0.001
```
`benchmark_out/report.md`, `results.json`, `roc_<model>.csv` পাবেন: প্রতি মডেলের EER, TAR@FAR (1e-2/1e-3/1e-4), 1:1 verification ও 1:N rank-1/open-set আলাদা, condition-ভিত্তিক ভাঙন (কোন অবস্থা সবচেয়ে খারাপ), fusion (mean-z) বনাম একক মডেল, CPU সময়। ডেটা কম হলে রিপোর্টে বড় করে "INSUFFICIENT DATA" লেখে এবং `calibrate.py` `calibrated:true` বসায় না।
ডিটেক্টর তুলনা, alignment পদ্ধতি তুলনা ও ফোনে RAM/সময় মাপা এই টুলে স্বয়ংক্রিয় নেই — ব্রাউজারে মাপতে হবে (**বাকি কাজ**)।

## Preprocessing সহনসীমা
`tests/js/parity.test.mjs` ও `tests/reference/make_reference.py` দেখুন (warp-only ≥ 0.999, e2e ≥ 0.97, ল্যান্ডমার্ক ≤ ২% চোখ-দূরত্ব)।
কোয়ালিটি থ্রেশহোল্ডও (`quality` in manifest) একই কারণে প্রাথমিক; আপনার ক্যামেরার ছবি থেকে ক্যালিব্রেট করুন।
