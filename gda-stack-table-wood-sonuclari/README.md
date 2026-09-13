# GDA Stack / Table-Wood deney sonuçları

Bu klasör, **Generalizable Domain Adaptation for Sim-and-Real Policy
Co-Training** resmî kodunun `Stack_RL2_range` görevindeki tek tohumlu kontrollü
yeniden üretim sonuçlarını içerir.

- Kaynak görsel alan: `rgb`
- Hedef görsel alan: `table-wood`
- Eğitim tohumu: `1`
- Eğitim bütçesi: yöntem başına 500 epoch × 300 adım = 150.000 güncelleme
- Ana final test: epoch500, yöntem başına 50 down + 50 up/OOD rollout
- Checkpoint taraması: epoch100/200/300/400/500, hücre başına 10+10 geliştirme rollout'u

## Ana epoch500 final sonucu

| Yöntem | Down | Up/OOD | Toplam |
|---|---:|---:|---:|
| **OT-Sim2Real** | **32/50 (%64)** | **20/50 (%40)** | **52/100** |
| Co-training | 29/50 (%58) | 19/50 (%38) | 48/100 |
| MMD | 21/50 (%42) | 9/50 (%18) | 30/100 |
| Target-only | 9/50 (%18) | 0/50 (%0) | 9/100 |
| Source-only | 0/50 (%0) | 0/50 (%0) | 0/100 |

OT-Sim2Real sayısal olarak birinci oldu. MMD ve tek alan baseline'larına göre fark
belirgindir. Co-training ile fark daha küçüktür: toplamda 4, Up/OOD alanında yalnız
1 ek başarılı rollout. Tek eğitim tohumu nedeniyle OT'nin co-training'e kesin
üstünlüğü iddia edilmez.

## Checkpoint duyarlılık sonucu

| Yöntem | Seçilen epoch | Down | Up/OOD | Toplam |
|---|---:|---:|---:|---:|
| **Co-training** | **200** | 5/10 | 7/10 | **12/20** |
| **OT-Sim2Real** | **400** | 7/10 | 3/10 | **10/20** |
| **MMD** | **500** | 5/10 | 2/10 | **7/20** |
| **Target-only** | **300** | 3/10 | 0/10 | **3/20** |
| **Source-only** | **500*** | 0/10 | 0/10 | **0/20** |

`*` Source-only bütün checkpointlerde 0/20 aldı; eşitlik kuralı en ileri epochu
seçti. Seçim kuralı toplam başarı, eşitlikte Up/OOD başarısı, sonra ileri epochtur.

Checkpoint taraması geliştirme başlangıçlarını kullanır ve ana final testin yerine
geçmez. OT'nin beş checkpoint'i ile MMD ve co-training epoch500 sonuçları seçim
protokolü yazılırken biliniyordu; bu nedenle tarama kısmen post-hoc ikincil analizdir.

## Yöntemler

- **OT-Sim2Real:** 1000 kaynak ve 10 hedef gösterimini kullanır. Davranış klonlama
  kaybına optimal taşıma tabanlı temsil eşleştirme kaybı ekler.
- **MMD:** Aynı verileri kullanır; kaynak ve hedef özellik dağılımlarını
  MMD/Energy kaybıyla yaklaştırır.
- **Co-training:** Aynı kaynak ve hedef verilerini birlikte eğitir; özel alan
  uyarlama kaybı kullanmaz.
- **Source-only:** Yalnız 1000 kaynak/simülasyon gösterimini kullanır.
- **Target-only:** Yalnız 10 hedef/down gösterimini kullanır.

## Dosyalar

- [`GDA-MAKALE-AKISI-VE-YONTEMLER.md`](GDA-MAKALE-AKISI-VE-YONTEMLER.md):
  makaledeki veri–eğitim–değerlendirme akışı, Mermaid diyagramları, yöntem
  karşılaştırmaları ve kısaltmalar
- [`ana-final-sonuclari.json`](ana-final-sonuclari.json): 500-rollout ana sonuç
- [`ana-final-sonuclari.csv`](ana-final-sonuclari.csv): ana sonucun tablo biçimi
- [`checkpoint-duyarlilik-sonuclari.json`](checkpoint-duyarlilik-sonuclari.json):
  25 checkpoint hücresi ve seçilen epochlar
- [`checkpoint-duyarlilik-sonuclari.csv`](checkpoint-duyarlilik-sonuclari.csv):
  tam checkpoint matrisi
- [`SHA256SUMS`](SHA256SUMS): yayımlanan sonuç ve açıklama dosyalarının bütünlük
  hash'leri

## Yorum sınırları

Bu çalışma tek görev ve tek eğitim tohumu içerir. Rolloutlar aynı eğitilmiş politika
içindeki başlangıç değişkenliğini ölçer; eğitimler arası değişkenliği ölçmez. Deney
fiziksel robotta yapılmadı ve gerçek dünya sim2real sonucu olarak sunulamaz.
