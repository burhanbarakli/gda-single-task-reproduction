# GDA makale akışı ve yöntemler

Bu belge, **Generalizable Domain Adaptation for Sim-and-Real Policy
Co-Training** makalesindeki ana yöntemin veri akışını, kayıplarını ve karşılaştırma
yöntemlerini açıklar. Makaledeki ana yöntem deney tablolarında **Ours**, resmî
proje ve bu yeniden üretimde **OT-Sim2Real** adıyla anılır.

## Ana yöntem akışı

```mermaid
flowchart TB
    SRC["Kaynak alan — Dsrc<br/>Çok sayıda simülasyon gösterimi"]
    TGT["Hedef alan — Dtgt<br/>Az sayıda gerçek dünya gösterimi"]

    subgraph OTBRANCH["OT / UOT HİZALAMA KOLU"]
        direction TB
        DTW["DTW — Dynamic Time Warping<br/>Yörüngelerin zamansal benzerliğini hesapla"]
        DTW --> TAS["Temporally Aligned Sampling<br/>Benzer aşamalardaki kaynak–hedef<br/>yörünge çiftlerini örnekle"]
        COST["Ortak maliyet matrisi C<br/>Görsel özellik uzaklığı<br/>+ proprioception uzaklığı"]
        COST --> UOT["UOT — Unbalanced Optimal Transport<br/>Sinkhorn–Knopp ile taşıma planı Π*<br/>Eşleşmeyen örnekleri zorla eşleştirmez"]
        UOT --> LOT["Hizalama kaybı<br/>L-UOT = ⟨Π*, C⟩"]
    end

    subgraph BCBRANCH["BC / POLİTİKA ÖĞRENME KOLU"]
        direction TB
        BC["BC mini-batch<br/>Kaynak + hedef uzman gösterimleri"]
        DP["Diffusion Policy / UNet<br/>Gürültülü eylem dizisinden<br/>eklenen gürültüyü tahmin et"]
        DP --> LBC["Davranış klonlama kaybı<br/>L-BC = MSE(ε̂, ε)"]
    end

    SRC --> DTW
    TGT --> DTW
    SRC --> BC
    TGT --> BC

    TAS --> ENC["Paylaşılan görsel kodlayıcı fφ<br/>ResNet18: görüntü o → gizli özellik z"]
    BC --> ENC
    ENC --> COST
    ENC --> DP

    LOT --> TOTAL
    LBC --> TOTAL
    TOTAL["Toplam kayıp<br/>L-total = L-BC + λ L-UOT"]
    TOTAL --> UPDATE["Kodlayıcı fφ ve politika πθ<br/>birlikte güncellenir"]

    UPDATE --> ID["Target / ID testi<br/>Hedefte eğitimdekine benzer durumlar"]
    UPDATE --> OOD["Target-OOD testi<br/>Hedef gösterimlerinde bulunmayan<br/>ama simülasyonda görülen durumlar"]

    classDef data fill:#e8f3ff,stroke:#2574a9,color:#111;
    classDef align fill:#fff0d9,stroke:#d17b0f,color:#111;
    classDef learn fill:#e7f7ec,stroke:#318a4f,color:#111;
    classDef test fill:#f3e9ff,stroke:#7651a8,color:#111;
    class SRC,TGT data;
    class DTW,TAS,COST,UOT,LOT align;
    class ENC,BC,DP,LBC,TOTAL,UPDATE learn;
    class ID,OOD test;
    style OTBRANCH fill:#fff8ed,stroke:#d17b0f,stroke-width:3px,stroke-dasharray:8 5,color:#8a4a00;
    style BCBRANCH fill:#effaf2,stroke:#318a4f,stroke-width:3px,stroke-dasharray:8 5,color:#205f36;
```

Dıştaki turuncu kesikli **OT / UOT hizalama kolu**, toplam amaç fonksiyonundaki
`L_UOT` terimini üretir. Yeşil kesikli **BC / Politika Öğrenme Kolu**, robotun
uzman eylemlerini taklit etmesini sağlayan `L_BC` terimini üretir. İki kol aynı
görsel kodlayıcıyı günceller ve `L_total` kutusunda birleşir.

Akışın temel adımları şöyledir:

1. Kaynak veri simülasyondan bol miktarda, hedef veri gerçek robottan az sayıda
   elde edilir.
2. **DTW**, davranış olarak birbirine benzeyen kaynak ve hedef yörüngelerini
   belirler.
3. Zamansal örnekleyici, görevin benzer aşamalarındaki kaynak ve hedef geçişlerini
   aynı mini-batch içine getirir.
4. Ortak görsel kodlayıcı iki alanın gözlemlerini aynı gizli uzaya taşır.
5. **UOT**, benzer proprioceptive durumlara ait görsel özellikleri hizalar. Veri
   sayıları dengesiz olduğundan bütün örnekleri zorla eşleştirmez.
6. **Behavior Cloning**, uzman eylemlerini öğrenir; hizalama kaybı kodlayıcının
   görsel alan değişimine dayanıklı özellikler üretmesini teşvik eder.
7. Kodlayıcı ve politika aşağıdaki ortak amaçla birlikte güncellenir:

   `L_total = L_BC + λ × L_UOT`

8. Politika hedef alanın eğitimdekine benzer durumlarında ve hedef
   gösterimlerinde bulunmayan **OOD** durumlarında değerlendirilir.

Makalenin kuramsal anlatımı görsel özellik–eylem çiftlerinin ortak dağılımını
hizalar. Pratik uygulama, simülasyon ve gerçek robot eylem temsilleri arasındaki
farklardan etkilenmemek için eylem uzaklığı yerine robotun proprioceptive durumunu
kullanır.

## BC / politika öğrenme kolunda ne çalışır?

**BC — Behavior Cloning (Davranış Klonlama)**, robotun görevi gerçekleştirmek
için hangi hareketleri yapacağını öğrenen ana politika eğitimidir. Bu çalışma
pekiştirmeli öğrenme değildir; ödül sinyali veya çevrim içi keşif kullanmaz.

1. Uzman gösteriminden görüntü `o`, robotun proprioceptive durumu `x` ve gerçek
   eylem dizisi `a` alınır.
2. ResNet18 görsel kodlayıcı görüntüyü gizli özellik `z = fφ(o)` biçimine getirir.
3. Eğitim sırasında uzman eylem dizisine rastgele Gauss gürültüsü `ε` eklenir.
4. Diffusion Policy'nin UNet modeli, `z`, `x` ve difüzyon zaman adımını kullanarak
   eklenen gürültüyü `ε̂` olarak tahmin eder.
5. Davranış klonlama kaybı `L_BC = MSE(ε̂, ε)` ile hesaplanır. Bu kayıp hem
   politika ağını hem de paylaşılan görsel kodlayıcıyı günceller.
6. Rollout sırasında uzman eylemi, BC kaybı, DTW veya UOT hesaplanmaz. Eğitilmiş
   kodlayıcı ve Diffusion Policy, rastgele gürültüyü yinelemeli olarak temizleyip
   uygulanacak eylem dizisini üretir.

OT kolu aynı davranış aşamasına ait kaynak ve hedef görüntülerinin benzer özellik
üretmesini sağlar. BC kolu bu özelliklerden Stack görevini gerçekleştirecek eylemi
öğrenir. OT kolu kaldırılıp yalnız kaynak ve hedef BC kaybı bırakıldığında yöntem
**Co-training** olur.

## Karşılaştırılan yöntemler

```mermaid
flowchart LR
    A["Eğitim yöntemleri"] --> SO["Source-only<br/>Yalnız kaynak / sim verisi<br/>BC"]
    A --> TO["Target-only<br/>Yalnız az hedef / gerçek veri<br/>BC"]
    A --> CO["Co-training<br/>Kaynak + hedef verisi<br/>BC, açık hizalama yok"]
    A --> MM["MMD<br/>Kaynak + hedef + marjinal<br/>özellik dağılımı hizalama"]
    A --> OT["OT-Sim2Real / Ours<br/>Kaynak + hedef + DTW örnekleme<br/>+ ortak özellik–durum UOT hizalaması"]

    SO --> Q1["Simülasyondan doğrudan<br/>aktarımı ölçer"]
    TO --> Q2["Az hedef verisinin tek başına<br/>yeterliliğini ölçer"]
    CO --> Q3["Verileri karıştırmanın<br/>kazancını ölçer"]
    MM --> Q4["Genel dağılım hizalamanın<br/>kazancını ölçer"]
    OT --> Q5["Yapısal ve davranışa duyarlı<br/>hizalamanın kazancını ölçer"]

    classDef base fill:#edf1f5,stroke:#66717d,color:#111;
    classDef main fill:#ffe8bc,stroke:#ca7200,color:#111,stroke-width:3px;
    class SO,TO,CO,MM base;
    class OT main;
```

| Yöntem | Eğitim verisi | Amaç / kayıp | Sorduğu soru |
|---|---|---|---|
| **Source-only** | Yalnız kaynak | BC | Simülasyon politikası doğrudan hedefe aktarılabilir mi? |
| **Target-only** | Yalnız az hedef verisi | BC | Az sayıdaki hedef gösterimi tek başına yeterli mi? |
| **Co-training** | Kaynak + hedef | BC | İki veri kümesini karıştırmak ne kazandırır? |
| **MMD** | Kaynak + hedef | BC + marjinal özellik hizalama | Genel dağılım hizalama yararlı mı? |
| **OT-Sim2Real / Ours** | Kaynak + hedef | BC + DTW destekli UOT | Yapısal ve davranışa duyarlı hizalama yararlı mı? |

MMD ile ana yöntem arasındaki temel fark, MMD'nin kaynak ve hedef özellik
dağılımlarını genel olarak yakınlaştırmasıdır. OT-Sim2Real hangi örneklerin
birbiriyle eşleşeceğini bir taşıma planıyla belirler ve bu eşleşmeyi robot durum
bilgisiyle yönlendirir.

## Kısaltmalar ve açılımları

| Kısaltma | İngilizce açılımı | Türkçe karşılığı / işlevi |
|---|---|---|
| **GDA** | Generalizable Domain Adaptation | Genellenebilir alan uyarlama |
| **BC** | Behavior Cloning | Davranış klonlama |
| **DP** | Diffusion Policy | Difüzyon tabanlı eylem politikası |
| **OT** | Optimal Transport | Optimal taşıma |
| **UOT** | Unbalanced Optimal Transport | Dengesiz optimal taşıma |
| **DTW** | Dynamic Time Warping | Dinamik zaman bükme / yörünge hizalama |
| **MMD** | Maximum Mean Discrepancy | Maksimum ortalama ayrışması |
| **OOD** | Out-of-Distribution | Eğitim dağılımı dışındaki durum |
| **ID** | In-Distribution | Eğitim dağılımına benzer durum |
| **KL** | Kullback–Leibler Divergence | UOT marjinallerini esnekleştiren dağılım farkı |

Resmî depoda `diffusion_policy_MMD` adıyla sunulan karşılaştırma yolu,
`SamplesLoss("energy")` kullanır. Bu nedenle bu yeniden üretimdeki **MMD** etiketi
makale ve resmî depo adlandırmasını izler; uygulanan hizalama terimi Energy
Distance'tır.

## Bu yeniden üretimdeki karşılığı

Makalenin genel akışında kaynak alan simülasyon, hedef alan gerçek dünyadır. Bu
yeniden üretim fiziksel robot kullanmadı; kontrollü bir simülasyondan simülasyona
görsel alan değişimi kullandı:

```text
Kaynak: normal RGB simülasyonu
   ├── down: 500 gösterim
   └── up:   500 gösterim

Hedef: table-wood simülasyonu
   ├── down: 10 eğitim gösterimi
   └── up:   0 eğitim gösterimi → ana OOD/genelleme sınaması
```

Bu nedenle `up` sonucu, hedef görünümde eğitim gösterimi bulunmayan fakat kaynak
simülasyonda öğrenilen duruma genellemeyi ölçer. Fiziksel robot sim2real sonucu
değildir.

## Kaynaklar

- [GDA makalesi — arXiv:2509.18631](https://arxiv.org/abs/2509.18631)
- [Resmî proje sayfası](https://ot-sim2real.github.io/)
- [Resmî kod — GaTech-RL2/ot-sim2real](https://github.com/GaTech-RL2/ot-sim2real)
- [Bu yeniden üretimin protokolü](../docs/PROTOCOL.md)
- [Ana sonuçlar](README.md)
