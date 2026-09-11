# Deney notları dizini

Bu klasör, `Stack_RL2_range` ve `rgb → table-wood` tek görev yeniden üretimindeki beş eğitimin kısa, taşınabilir kayıtlarını içerir.

| Yöntem | Koşu | Durum | Geliştirme testi |
|---|---|---|---:|
| OT-Sim2Real | `g5_ot-sim2real_seed1_02` | 500 epoch tamamlandı | 9/20 (%45) |
| MMD | `g5_MMD_seed1_01` | 500 epoch tamamlandı | 7/20 (%35) |
| Co-training | `g5_cotrain_seed1_01` | 500 epoch tamamlandı | 8/20 (%40) |
| Source-only | `g5_source_only_seed1_01` | 500 epoch tamamlandı | yapılmadı |
| Target-only | `g5_target_only_seed1_01` | 500 epoch tamamlandı | yapılmadı |

Geliştirme değerleri iki kamera görünümünde 10'ar rollout ile alınmıştır; nihai karşılaştırma değildir. Nihai test her yöntem için 100 rollout kullanır.

