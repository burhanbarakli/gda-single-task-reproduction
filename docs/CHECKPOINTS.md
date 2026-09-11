# Checkpoint saklama ve taşıma

Her koşu `model_epoch_50.pth` ile `model_epoch_500.pth` arasında on periyodik checkpoint üretir. Nihai karşılaştırma yalnız `model_epoch_500.pth` dosyasını kullanır.

Checkpointler yaklaşık 1,26 GB olduğu için normal Git deposuna eklenmez. Dosyayı okul bilgisayarına taşırken `manifests/checkpoints.json` içindeki SHA-256 değerini doğrulayın:

```bash
sha256sum /path/to/model_epoch_500.pth
```

Yeni eğitim çıktılarından taşınabilir bir kayıt oluşturmak için:

```bash
python scripts/export_checkpoints.py --runs-root ./runs --output ./artifacts/gda-checkpoints
```

Bu komut checkpointleri kopyalar ve üretilen `manifest.json` içine kaynak yol, boyut ve SHA-256 ekler. `artifacts/` Git tarafından dışlanır; arşivi Drive veya kurumsal depolama ile taşıyın.

