# 🚀 AI/ML Integration Guide: Bridging Logic to UI

Halo Tim AI/ML! Repo ini adalah referensi standar untuk proyek **"Service Metabot/Generative UI"**. Inti dari panduan ini adalah memastikan bahwa setiap reasoning dan tool calling yang kalian buat di Python bisa tampil secara *seamless* dan interaktif di Web Playground kita.

Kalian cukup fokus mendalami dua file utama: `index.py` dan `stream.py`.

---

### 1. File `index.py` (The Gateway)

File ini adalah pintu masuk utama. Hal penting yang perlu kalian pahami di sini:

* **Protocol Handshake:** Kita menggunakan `protocol: str = Query('data')` untuk mencocokkan standar Vercel AI SDK.
* **Response Headers:** Fungsi `patch_response_with_headers` sangat krusial. Tanpa header ini, Frontend tidak akan tahu cara membedakan mana teks biasa dan mana instruksi visual (grafik/aksi).

---

### 2. File `stream.py` (The Heart of Communication)

Di sinilah keajaiban terjadi. AI tidak hanya mengirim teks, tapi mengirimkan **Events** menggunakan standar **Data Stream Protocol (DSP)**.

Pahami bagian ini agar UI kita tidak "blank" saat agen kalian sedang berpikir:

* **text-delta:** Kirimkan potongan teks agar user bisa melihat efek mengetik secara *real-time*.
* **tool-input-start:** Kirimkan ini saat agen mulai memanggil fungsi (*tool*). UI akan otomatis menampilkan status "Thinking" atau "Searching".
* **tool-output-available:** Ini bagian paling penting bagi tim BI. Kirimkan JSON hasil olahan data kalian di sini agar Frontend bisa merender Recharts secara otomatis.

---

### Mengapa Ini Penting?

Mentor kita mencanangkan visi **Chatbot Metabot**. Agar layanan AI yang kalian buat bisa sesuai dengan layanan tersebut, komunikasi agen wajib mematuhi protokol **Vercel Data Stream Protocol** yang ada di repo ini. Dengan mengikuti standar ini, Layanan kalian otomatis punya **Web Playground** yang cantik.

---

> **Selamat bereksperimen!** Jika ada perubahan logika pada model, pastikan format stream-nya tetap mengikuti standar di `stream.py` ya.