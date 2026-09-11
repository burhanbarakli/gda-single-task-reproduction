# Deney 1 — OT-Sim2Real

- Koşu: `g5_ot-sim2real_seed1_02`
- Config: `config/reference/ot-sim2real_seed1.json`
- Eğitim: 500 epoch × 300 adım, batch 128, seed 1
- Veri: 500 kaynak eşli + 500 kaynak eşsiz + 10 hedef eşli gösterim
- Uyum terimi: dengesiz optimal taşıma; DTW eşleştirme bilgisi kullanır
- Observation I/O: resmî yol
- Süre: 112.776,31 saniye (31,33 saat)
- Tepe GPU: 5.272 MiB allocated, 6.318 MiB reserved
- Epoch 500 checkpoint SHA-256: `1713ce826aa983ac82dffd63dd260b70a6797febe003d7d1cb93dd1a6261039b`
- Geliştirme testi: down 6/10, up 3/10, toplam 9/20 (%45)

Bu koşu resmî yönteme en yakın ana deneydir. İlk OT denemesi Windows GPU TDR nedeniyle başarısız olmuş, yeni koşu kimliğiyle baştan yürütülmüştür. Başarısız koşu performans karşılaştırmasına alınmaz.

