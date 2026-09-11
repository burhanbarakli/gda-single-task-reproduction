# Deney 3 — Co-training

- Koşu: `g5_cotrain_seed1_01`
- Config: `config/reference/cotrain_seed1.json`
- Eğitim: 500 epoch × 300 adım, batch 128, seed 1
- Veri hakları: OT ve MMD ile aynı
- Uyum terimi: yok; ağırlıklı kaynak ve hedef davranış klonlama
- Observation I/O: doğrulanmış `prefix-v1`
- Süre: 26.121,16 saniye (7,26 saat)
- Tepe GPU: 4.100 MiB allocated, 4.818 MiB reserved
- Epoch 500 checkpoint SHA-256: `9e2322c914e3299138be4641fd2128aadbeeb7c992cf5d92978596ae25998375`
- Geliştirme testi: down 5/10, up 3/10, toplam 8/20 (%40)

Bu baseline, aynı veriyi alan uyarlama kaybı olmadan birlikte eğitir. Böylece OT veya MMD yardımcı kaybının ek katkısını ayırmaya yarar.

