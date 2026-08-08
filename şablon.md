# 🧠 BrainHealthDetection (BHD) — YOLOv8 Beyin Tümörü Tespiti

Bu proje, **Ultralytics YOLOv8** kullanarak **MR/CT görüntülerinde beyin tümörü tespiti**, **otomatik etiketleme**, **dataset hazırlama** ve **fine-tuning** işlemlerini uçtan uca gerçekleştirir.

---

## 📘 Proje Yapısı

```
BrainHealthDetection/
├── brain/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
│
├── imagesfortuning/              # Etiketlenecek yeni görseller
├── runs/                         # YOLO eğitim çıktıları (ör. best.pt)
│   └── detect/train/weights/best.pt
│
├── dataset_tuning/               # Split_Dataset() sonucu oluşan dataset
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── valid/
│       ├── images/
│       └── labels/
│
├── predict                       # İsteğe bağlı tahmin sonuçlarının kaydedildiği klasör
│
├── BrainModel.py                 # YOLOv8 model eğitim dosyası
├── BrainPred.py                  # Otomatik etiketleme + fine-tuning pipeline
├── data.yaml                     # Ana dataset konfigürasyonu
├── data_tuning.yaml              # Fine-tuning dataset konfigürasyonu
├── LICENSE
├── requirements.txt
└── README.md
```

---

## ⚙️ Gereksinimler

Projeyi çalıştırmadan önce bağımlılıkları yükleyin ve sanal ortamı (virtual environment) aktif edin:

```bash
pip install -r requirements.txt
```

---

## 🚀 1️⃣ BrainModel.py — Model Eğitimi

`BrainModel.py`, YOLOv8 tabanlı beyin tümörü tespiti için temel modeli eğitir.

### 🧩 Özellikler

* CUDA kontrolü yapar (GPU varsa otomatik kullanır).
* `data.yaml` dosyasını kullanarak modeli eğitir.
* Eğitim sonunda `runs/detect/train/weights/best.pt` dosyası oluşturulur.

### ▶️ Çalıştırma

```bash
python BrainModel.py
```

---

## 🧠 2️⃣ BrainPred.py — Otomatik Etiketleme, Split ve Fine-Tuning

Bu betik, eğitilmiş YOLOv8 modelini kullanarak:

1. Yeni MR/CT görsellerini otomatik etiketler (`.txt` formatında).
2. Verileri `train/valid` olarak böler.
3. İstenirse fine-tuning sürecini otomatik başlatır.

---

### 🔹 `MakeReady()`

Bu fonksiyon, belirtilen klasördeki görselleri modele tahmin ettirir ve sonuçları `.txt` etiket dosyası olarak kaydeder.

```python
def MakeReady(
    Path_weights: str = "runs/detect/train/weights/best.pt",
    Path_img: str ="imagesfortuning/",
    Output_dir: str ="imagesfortuning",
    Conf_thres: float = 0.50):
```

Fonksiyon çalıştırıldığında kullanıcıya sorar:

```
Görseller etiketlenmeye başlanıyor. Sonuçları görseller üzerinde görmek ister misiniz? (Y/n)
```

* `Y` → Etiketli görseller "predict" klasörüne kaydedilir.
* `N` → Sadece `.txt` dosyaları oluşturulur.

Etiketleme tamamlandığında tekrar sorar:

```
Görselleri eğitime uygun ayırmak ister misin? (Y/n)
```

* `Y` → `Split_Dataset()` fonksiyonu çalışır.
* `N` → Program sonlandırılır.

---

### 🔹 `Split_Dataset()`

Etiketli görselleri kullanarak `dataset_tuning/` klasörünü oluşturur.
Varsayılan olarak verilerin **%80'i train**, **%20'si validation** olarak ayrılır.

```python
def Split_Dataset(
    Path_img: str = "imagesfortuning/",
    Output_dir: str = "dataset_tuning",
    split_ratio: float = 0.8):
```

Sonrasında kullanıcıya sırasıyla şu sorular yöneltilir:

```
Eğitim için split edilen veriler silinsin mi? (Y/n)
Fine_Tuning işlemine başlatılsın mı? (Y/n)
```

Cevaba göre `Clear_imagesfortuning()` ve `YoloFineTuning()` fonksiyonları devreye girer.

---

### 🔹 `YoloFineTuning()`

Yeni oluşturulan dataset (`data_tuning.yaml`) ile kısa bir fine-tuning eğitimi başlatır.

```python
def YoloFineTuning(Path_weights: str = "runs/detect/train/weights/best.pt"):
```

CUDA mevcutsa otomatik olarak GPU kullanılır.
Eğitim tamamlandığında şu mesaj görünür:

```
✅ Model iyileştirme başarılı.
```

---

### 🔹 `Clear_imagesfortuning()`

Bu işlem, **split** işlemine dahil edilen tüm dosyaları `imagesfortuning` klasöründen kaldırır.
Veriler **dataset_tuning/** dizininde korunur.

---

### 🔹 `Clear_dataset_tuning()`

Fine-tuning sonrası tüm split edilmiş verileri siler.

---

## 📄 YAML Dosyaları

### `data.yaml`

```yaml
train: ../brain/train/images
val: ../brain/valid/images
test: ../brain/test/images

nc: 3
names: ['glioma', 'meningioma', 'pituitary']
```

### `data_tuning.yaml`

```yaml
train: ../dataset_tuning/train/images
val: ../dataset_tuning/valid/images

nc: 3
names: ['glioma', 'meningioma', 'pituitary']
```

---

## 🔁 Akış Özeti

1️⃣ `BrainModel.py` → YOLOv8 modelini eğitir.
2️⃣ `BrainPred.py` → Yeni görselleri etiketler.
3️⃣ `Split_Dataset()` → Verileri train/valid olarak böler.
4️⃣ `YoloFineTuning()` → Yeni dataset ile modeli iyileştirir.
5️⃣ `Clear_*()` → Klasör temizliği yapar.

---

## ⚠️ Hata Durumları

| Durum                                      | Açıklama                                       |
| ------------------------------------------ | ---------------------------------------------- |
| `⚠️ CUDA mevcut değil.`                    | GPU bulunamadı, eğitim CPU üzerinde yapılır.   |
| `⚠️ UYARI: Etiket dosyası bulunamadı.`     | Görsellerden biri için etiket oluşturulamadı.  |
| `❌ Model eğitilemedi, eğitim başarısız.`   | BrainModelCreat() aşamasında model ana eğitim sırasında oluşan hataları kapsar. |
| `❌ Hata oluştu. Görseller etiketlenemedi.` | MakeReady() sırasında hata oluştu.             |
| `❌ Model iyileştirme başarısız oldu.`      | YoloFineTuning() eğitimi başarısız tamamlandı. |

---

## 📊 Model Eğitim Sonuçları (İlk 10 Epoch)

> [!NOTE]
> Bu eğitim sonuçları, sistemin çalışabilirliğini ve pipeline yapısını test etmek amacıyla başlatılmış **kısa süreli (10 epoch) bir deneme eğitimine aittir**. Eğitim devam ettirilmemiştir. Tam performans ve üretim aşaması için modelin daha yüksek epoch sayılarında (ör. 50-100) yeniden eğitilmesi önerilir.

İlk 10 epoch sonucunda elde edilen YOLOv8 model metrikleri oldukça başarılıdır. Aşırı öğrenme (overfitting) olmadan validation kayıplarının (loss) düştüğü gözlemlenmiştir.

* **mAP@50:** %87.91 (Genel doğruluk)
* **Precision:** %87.20 (Doğru tespit oranı)
* **Recall:** %78.90 (Tümörleri yakalama oranı)
* **mAP@50-95:** %62.39

Eğitim sürecindeki grafiksel sonuçlar (Confusion Matrix, PR eğrileri vb.) `runs/detect/train/` dizininden incelenebilir.

---

## 📈 Çıktılar

* Eğitim sonuçları: `runs/detect/train/weights/best.pt`
* Etiketli yeni veriler: `imagesfortuning/*.txt`
* Split edilmiş dataset: `dataset_tuning/`
* Fine-tuned model: `runs/detect/train/weights/last.pt` veya `best.pt`

---

## 👤 Yazar

**Erdem Oruç**
Computer Vision & AI Developer

> YOLOv8 ile beyin tümörü tespiti ve veri hazırlama pipeline çalışması.
