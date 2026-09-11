# GDA tek görev yeniden üretim paketi

Bu depo, **Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training** resmî kodunu tek bir simülasyon görevinde kontrollü biçimde yeniden üretir. Görev `Stack_RL2_range`, görsel hedef alan `table-wood`, kaynak alan `rgb` ve eğitim tohumu `1`'dir.

Beş eğitim 500 epoch tamamlandı: OT-Sim2Real, MMD, co-training, source-only ve target-only. Her yöntem 300 adım/epoch ile 150.000 güncelleme gördü. Bu deney fiziksel robot içermediği için sonuçlar simülasyondaki görsel alan değişimine aittir.

## Hızlı başlangıç

Ubuntu 24.04 veya Windows 11 üzerinde WSL2 + Ubuntu 24.04, NVIDIA sürücüsü ve en az 12 GiB boş GPU belleği önerilir.

```bash
git clone https://github.com/burhanbarakli/gda-single-task-reproduction.git
cd gda-single-task-reproduction
bash scripts/bootstrap_ubuntu.sh
source .venv/bin/activate
python scripts/download_official_dataset.py --output ./data/download
python scripts/prepare_dataset_layout.py --download-root ./data/download --data-root ./data/gda
python scripts/verify_dataset.py --data-root ./data/gda
python scripts/train_method.py --method target_only --data-root ./data/gda --output-root ./runs --validate-only
python scripts/train_method.py --method target_only --data-root ./data/gda --output-root ./runs
```

Beş yöntemi sırayla çalıştırmak için:

```bash
bash scripts/run_all.sh /path/to/gda-data ./runs
```

`--validate-only` veri hashlerini, kaynak commitlerini, patch'i, config'i ve GPU görünürlüğünü kontrol eder; eğitimi başlatmaz. OT en fazla GPU belleği ve süre kullanır. Koşuları aynı anda aynı GPU üzerinde başlatmayın.

## Rapor ve canlı takip

Tamamlanmış deneylerin çevrimdışı raporu [reports/gda-training-report.html](reports/gda-training-report.html) içindedir. Grafikler tarayıcıda JavaScript ile çizilir; internet bağlantısı gerekmez.

Her eğitim için hazırlanan kısa Markdown kayıtları hem [`docs/experiments/`](docs/experiments/) altında hem de [Google Drive deney notları klasöründe](https://drive.google.com/drive/folders/1vGsubzHN8jb1d4dNg6B4scuhYRNzRWx4) bulunur. Drive klasörü paylaşılmamıştır; bağlı Google hesabıyla açılır.

Yeni bir makinedeki koşuları canlı izlemek için:

```bash
python scripts/live_dashboard.py --runs-root ./runs --port 8765
```

Ardından `http://127.0.0.1:8765/` adresini açın. Tamamlanan loglardan raporu yenilemek için:

```bash
python scripts/build_report.py --source-root . --output reports/gda-training-report.html
```

## Veri ve checkpointler

2,41 GB resmî veri ve yaklaşık 1,26 GB büyüklüğündeki checkpointler GitHub'a eklenmez. [manifests/datasets.json](manifests/datasets.json) her veri dosyasının beklenen yolunu, boyutunu ve SHA-256 değerini; [manifests/checkpoints.json](manifests/checkpoints.json) tamamlanmış modellerin kaydını içerir.

Bir checkpoint arşivi hazırlamak için:

```bash
python scripts/export_checkpoints.py --runs-root ./runs --output ./artifacts/gda-checkpoints
```

## Protokol sınırı

Küçük 20-rollout kontrolleri geliştirme testidir. Nihai karşılaştırma, epoch 500 checkpointleri üzerinde her yöntem için iki kamera görünümünde 50'şer, toplam 100 rollout kullanır. Ayrıntılar [docs/PROTOCOL.md](docs/PROTOCOL.md) ve yöntem notlarında bulunur.

## Kaynak

- Resmî kod: [GaTech-RL2/ot-sim2real](https://github.com/GaTech-RL2/ot-sim2real), commit `114704b6b381b410cc14a51cb16952d1f4e8c69d`
- Lisans: MIT; üçüncü taraf bileşenlerin kendi lisansları geçerlidir.
