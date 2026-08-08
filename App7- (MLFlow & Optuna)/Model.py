import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import optuna

# 1. Veriyi yükleme ve Ön İşleme
df = pd.read_csv("disease_prediction.csv")
df = df.drop("patient_id", axis=1)

binary_cols = ['smoking', 'alcohol_consumption', 'family_history', 'disease']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

df = pd.get_dummies(df, columns=['gender', 'physical_activity'], drop_first=True)

X = df.drop("disease", axis=1)
y = df["disease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MLFLOW & OPTUNA KISMI ---

# 1. Kayıtların Nereye Yapılacağını KESİN OLARAK Belirliyoruz
# Hata tamamen benden kaynaklı! Kayıt yerini kodda belirtmeyi unuttum.
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# MLflow için deney (experiment) ismini belirliyoruz
mlflow.set_experiment(f"Disease_Prediction_RF_Optimization_V2")

def objective(trial):
    # MLflow ile iç içe kayıt işlemi (nested=True) başlatıyoruz
    with mlflow.start_run(nested=True,run_name=f"trial_TEST_{trial.number}"):
        
        # 1. Denenecek Parametreleri ve Aralıklarını Belirliyoruz
        n_estimators = trial.suggest_int('n_estimators', 1, 100)
        max_depth = trial.suggest_int('max_depth', 1, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        
        # 2. Modeli Optuna'nın o an önerdiği parametrelerle kuruyoruz
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=42
        )
        
        # 3. Model Eğitimi (Evet, asıl eğitim tam da burada, MLflow'un gözleri önünde oluyor!)
        rf.fit(X_train, y_train)
        
        # 4. Tahmin ve Başarı Hesaplama
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 5. MLflow'a Kayıt İşlemleri
        # Optuna'nın denediği parametreleri kaydet:
        mlflow.log_params(trial.params)
        # Elde edilen başarı skorunu kaydet:
        mlflow.log_metric("accuracy", accuracy)
        
        # Optuna'nın anlayabilmesi için sonucu geriye döndürüyoruz
        return accuracy

# --- Ana Çalışmayı (Study) Başlatma ---
if __name__ == "__main__":
    print("MLflow ve Optuna optimizasyonu başlıyor... Lütfen bekleyin.")
    
    # Ana bir MLflow run'ı açıyoruz (Optuna'nın denemeleri bunun içine 'nested' olarak girecek)
    with mlflow.start_run(run_name="RandomForest_Optuna_Study"):
        
        # Optuna'ya "amacımız accuracy (başarı) değerini maksimize etmek (maximize)" diyoruz
        study = optuna.create_study(direction="maximize")
        
        # Optuna'ya "objective fonksiyonunu al ve 10 farklı kombinasyon dene" diyoruz (Öğrenme amaçlı 10 ideal)
        study.optimize(objective, n_trials=50)
        
        print("\n--- Optimizasyon Tamamlandı! ---")
        print(f"En yüksek başarı oranı (Accuracy): {study.best_value:.4f}")
        print("En iyi sonucu veren parametreler:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
            
    print("\nArtık terminale 'mlflow ui' yazarak sonuçları tarayıcıda görebilirsin!")
