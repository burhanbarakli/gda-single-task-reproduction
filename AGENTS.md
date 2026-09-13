# Çalışma kuralları

Bu depo, Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training yönteminin tek resmî simülasyon görevindeki kontrollü yeniden üretim paketidir.

- Resmî `GaTech-RL2/ot-sim2real` kodunu `114704b6b381b410cc14a51cb16952d1f4e8c69d` commit'inde kullan.
- Beş yöntem, `Stack_RL2_range`, `table-wood`, seed 1 ve donmuş veri hakları korunur.
- Sonuçlara baktıktan sonra seed, veri bölümü veya test resetlerini değiştirme.
- Eğitim sırasında rollout yapma. Geliştirme testlerini nihai testten ayrı tut.
- Fiziksel robot kullanılmadığı için sonucu gerçek sim2real başarısı olarak adlandırma.
- `data/`, `runs/`, `artifacts/` ve büyük checkpointler Git'e eklenmez. Kayıtları SHA-256 ile doğrula.

## Sonraki deney worktree'si

Bu worktree `codex/next-experiments` dalındaysa tamamlanmış yeniden üretim,
`gda-stack-table-wood-seed1-final` etiketi ve
`01aecc2b094400258d1b7afeb8f63b9c31064899` commit'iyle dondurulmuştur.
Yeni betik, config, küçük log ve sonuçları `experiments-next/` altında ayrı deney
klasörlerine yaz. `gda-stack-table-wood-sonuclari/`, `results/`, `reports/` ve
mevcut manifestleri tarihsel kanıt olarak koru. Yeni sonuçları eski G5 sonucu gibi
sunma ve kullanıcı açıkça istemeden `main` dalına birleştirme.
