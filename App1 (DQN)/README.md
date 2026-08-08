# 🧠 App1 (DQN) — CartPole Deep Q-Network

Bu proje, **Stable Baselines3 (SB3)** kullanılarak **CartPole-v1 ortamında DQN (Deep Q-Network)** ajanının eğitilmesi, **test edilmesi** ve **MLFlow ile loglanması** işlemlerini uçtan uca gerçekleştirir.

---

## 📘 Proje Yapısı

```
App1 (DQN)/
├── DqnModel.py                 # DQN model eğitim ve hiperparametre optimizasyon dosyası
├── test_model.py               # Eğitilmiş modeli test etme dosyası
├── baslat_mlflow.bat           # MLFlow arayüzünü başlatan script
├── mlruns/                     # MLFlow logları ve model çıktıları
├── *.zip                       # Eğitilmiş model dosyaları
└── README.md
```

---

## 🚀 1️⃣ DqnModel.py — Model Eğitimi ve Optimizasyon

`DqnModel.py`, CartPole-v1 ortamı için DQN ajanını eğitir.

### 🧩 Özellikler

* Optuna kullanılarak hiperparametre optimizasyonu yapar.
* MLFlow ile eğitim metriklerini (loss, reward vb.) otomatik takip eder.
* Eğitim sonunda en iyi modeli kaydeder.

### ▶️ Çalıştırma

```bash
python DqnModel.py
```

---

## 🧠 2️⃣ test_model.py — Model Testi

Bu betik, eğitilmiş modeli kullanarak çevreyi görselleştirir ve ajanın performansını test eder.

### ▶️ Çalıştırma

```bash
python test_model.py
```

---

## 📈 MLFlow ile Takip

Eğitim sürecindeki grafikleri ve metrikleri incelemek için `baslat_mlflow.bat` dosyasını çalıştırın veya terminalden şu komutu girin:

```bash
mlflow ui
```

## 👤 Yazar

**Erdem Oruç**
