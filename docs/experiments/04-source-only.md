# Deney 4 — Source-only

- Koşu: `g5_source_only_seed1_01`
- Config: `config/reference/source_only_seed1.json`
- Eğitim: 500 epoch × 300 adım, batch 128, seed 1
- Veri: 500 kaynak eşli + 500 kaynak eşsiz; hedef gösterim yok
- Uyum terimi: yok
- Observation I/O: doğrulanmış `prefix-v1`
- Süre: 46.471,73 saniye (12,91 saat)
- Tepe GPU: 3.490 MiB allocated, 4.208 MiB reserved
- Epoch 500 checkpoint SHA-256: `888543ee2c2e09ce0cae27108151a9a8e9040ed4810135be9b53ada4c97f6697`
- Geliştirme testi: yürütülmedi

Bu baseline, hedef alan verisi olmadan görsel alan değişiminin oluşturduğu aktarım boşluğunu ölçmek içindir.

