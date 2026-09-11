# Deney 2 — MMD

- Koşu: `g5_MMD_seed1_01`
- Config: `config/reference/MMD_seed1.json`
- Eğitim: 500 epoch × 300 adım, batch 128, seed 1
- Veri hakları: OT ile aynı
- Uyum terimi: Maximum Mean Discrepancy
- Observation I/O: doğrulanmış `prefix-v1`
- Süre: 53.338,86 saniye (14,82 saat)
- Tepe GPU: 5.877 MiB allocated, 7.330 MiB reserved
- Epoch 500 checkpoint SHA-256: `d31da5b82fee05d33124ac97fe3a69514b4c55147653f473ae468e0a226cadc1`
- Geliştirme testi: down 5/10, up 2/10, toplam 7/20 (%35)

MMD, kaynak ve hedef temsil dağılımlarını çekirdek tabanlı bir uzaklıkla yaklaştırır. Loss ölçeği OT ile doğrudan karşılaştırılamaz; yardımcı terimlerin tanımı farklıdır.

