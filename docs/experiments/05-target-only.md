# Deney 5 — Target-only

- Koşu: `g5_target_only_seed1_01`
- Config: `config/reference/target_only_seed1.json`
- Eğitim: 500 epoch × 300 adım, batch 128, seed 1
- Veri: yalnız 10 hedef eşli gösterim
- Uyum terimi: yok
- Observation I/O: doğrulanmış `prefix-v1`
- Süre: 39.402,37 saniye (10,95 saat)
- Tepe GPU: 3.490 MiB allocated, 4.208 MiB reserved
- Epoch 500 checkpoint SHA-256: `7ca6df713aebee8f20fbb166466ffc217399ef12d645fb86185b95f9a14f3ffa`
- Geliştirme testi: yürütülmedi

Bu baseline, yalnız az sayıdaki hedef gösterimle eğitimin seviyesini ölçer. Daha önce hazırlanmış okul bilgisayarı paketiyle aynı hedef-veri hakkını kullanır.

