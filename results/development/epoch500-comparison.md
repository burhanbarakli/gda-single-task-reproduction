# OT–MMD–co-training epoch500 geliştirme karşılaştırması

Aynı eğitim tohumu ve her alan için aynı 10 başlangıç kullanıldı.
Bu ara ölçüm nihai G5 testi değildir.

| Alan | OT | MMD | Co-training |
|---|---:|---:|---:|
| down | 6/10 | 5/10 | 5/10 |
| up | 3/10 | 2/10 | 3/10 |
| birleşik | 9/20 | 7/20 | 8/20 |

Twenty fixed development rollouts per method are an early paired descriptive check. They do not replace the frozen 500-rollout final G5 comparison and are not used to select checkpoints, settings, or required baselines.
