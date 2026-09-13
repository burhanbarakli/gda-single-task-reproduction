# GDA sonraki deneyler çalışma alanı

Bu Git worktree, tamamlanmış `Stack_RL2_range / table-wood / seed1` yeniden
üretimini değiştirmeden yeni çalışmalara devam etmek için ayrılmıştır.

## Başlangıç noktası

- Dondurulmuş etiket: `gda-stack-table-wood-seed1-final`
- Dondurulmuş commit: `01aecc2b094400258d1b7afeb8f63b9c31064899`
- Devam dalı: `codex/next-experiments`
- Ana yeniden üretim dalı: `main`
- Ana yeniden üretim worktree'si:
  `C:\CODES\GDA\release\gda-single-task-reproduction`

## Ayrım kuralları

- Tamamlanmış sonuçlar, raporlar, manifestler ve protokol kayıtları tarihsel kanıt
  olarak korunur.
- Yeni deneylerin betik, config, küçük log ve sonuçları `experiments-next/`
  altında kendi adlandırılmış klasörlerine yazılır.
- `data/`, `runs/`, `artifacts/` ve büyük checkpointler Git'e eklenmez; mevcut
  resmî veri ve checkpointler tam yolları ve SHA-256 kimlikleriyle referanslanır.
- Yeni veri bölümü, tohum veya model seçimi eski G5 sonucunun parçası gibi
  sunulmaz.
- `main` dalına birleşim ancak kullanıcı açıkça istediğinde yapılır.

Bu ayrım Git düzeyindedir: yeni dalda yapılan commitler tamamlanmış yeniden üretim
etiketini veya `main` dalındaki yayımlanmış sonucu değiştirmez.
