# Donmuş deney protokolü

## Amaç

Resmî GDA kodundaki alan uyarlama yaklaşımını `Stack_RL2_range` görevinde, `rgb → table-wood` görsel alan değişiminde yeniden üretmek ve dört karşılaştırmayla aynı eğitim bütçesinde değerlendirmek.

## Ortak eğitim ayarları

| Alan | Değer |
|---|---:|
| Tohum | 1 |
| Epoch | 500 |
| Adım / epoch | 300 |
| Güncelleme / yöntem | 150.000 |
| Batch | 128 |
| Worker | 2 |
| Sequence length | 16 |
| Frame stack | 2 |
| Checkpoint | Her 50 epoch |
| Model | Diffusion Policy, ResNet18 + UNet |

Eğitim sırasında rollout kapalıdır. OT resmî observation I/O yolunu; diğer dört yöntem doğrulanmış `prefix-v1` okuma düzeltmesini kullanır. Bu fark raporda açıkça kayıtlıdır.

## Veri hakları

| Yöntem | Kaynak eşli | Kaynak eşsiz | Hedef eşli | Hedef eşsiz |
|---|---:|---:|---:|---:|
| OT-Sim2Real | 500 × 0,45 | 500 × 0,45 | 10 × 0,10 | 0 |
| MMD | 500 × 0,45 | 500 × 0,45 | 10 × 0,10 | 0 |
| Co-training | 500 × 0,45 | 500 × 0,45 | 10 × 0,10 | 0 |
| Source-only | 500 × 0,50 | 500 × 0,50 | 0 | 0 |
| Target-only | 0 | 0 | 10 × 1,00 | 0 |

## Test ayrımı

Geliştirme kontrolleri iki görünümde 10'ar rollout kullanır ve hata ayıklama içindir. Nihai test, sonuçlara bakılmadan dondurulan reset tohumlarıyla her yöntem ve görünüm için 50 rollout kullanır. Geliştirme sonuçları nihai yöntem sıralaması değildir.

