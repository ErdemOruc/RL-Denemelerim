# Gymnasium, Reinforcement Learning ortamlarını oluşturmak için kullanılan kütüphanedir (Eski adıyla OpenAI Gym).
import gymnasium as gym

# MLflow, deneylerimizi, hiperparametreleri ve sonuçları loglamak (kaydetmek) için kullanılır.
import mlflow

# os modülü, işletim sistemiyle etkileşim kurmak, çevre değişkenlerini ayarlamak için kullanılır.
import os

# numpy, matematiksel işlemler ve dizilerle çalışmak için kullanılır (ortalama almak vb.).
import numpy as np

# optuna, hiperparametre optimizasyonu (en iyi ayarları bulmak) için kullanılan kütüphanedir.
import optuna

# deque, baştan ve sondan hızlıca eleman ekleyip çıkarabildiğimiz özel bir liste türüdür. Son 100 skoru tutmak için kullanacağız.
from collections import deque

# Stable Baselines 3 kütüphanesinden DQN algoritmasını içe aktarıyoruz.
from stable_baselines3 import DQN

# Monitor, eğitim sırasında ortamdan her bölüm (episode) sonunda uzunluk ve skor gibi bilgileri almamızı sağlar.
from stable_baselines3.common.monitor import Monitor

# BaseCallback, modelin eğitimi sırasında belirli adımlarda (örneğin her adımda) araya girip kendi kodumuzu çalıştırmamızı sağlar.
from stable_baselines3.common.callbacks import BaseCallback

# Kendi özel geri arama (callback) sınıfımızı oluşturuyoruz, BaseCallback'ten miras alıyoruz.
class MLflowCallback(BaseCallback):
    """
    Özel Callback: Her bölüm bittiğinde MLflow'a metrikleri kaydeder ve
    ortalama skor 195'e ulaştığında eğitimi durdurur.
    """
    # Sınıf başlatılırken çalışacak olan fonksiyon
    def __init__(self, verbose=0):
        # Üst sınıfın (BaseCallback) init fonksiyonunu çağırıyoruz
        super().__init__(verbose)
        # Toplamda kaç bölüm (oyun) oynandığını saymak için bir değişken
        self.episode_count = 0
        # Sadece son 100 bölümün skorunu tutacak, 100'ü geçince en eskiyi silecek özel bir liste
        self.scores_window = deque(maxlen=100)
        # Erken durdurma (pruning) yapılıp yapılmadığını takip edeceğimiz değişken
        self.is_pruned = False

    # Bu fonksiyon, ajanın ortamda attığı her bir "adım" (step) sonrasında otomatik olarak çağrılır.
    def _on_step(self) -> bool:
        # self.locals içindeki 'infos' sözlüğü, bölüm (episode) bittiğinde ortamdan dönen bilgileri içerir
        for info in self.locals.get("infos", []):
            # Eğer info sözlüğünün içinde 'episode' diye bir anahtar varsa, bu demektir ki bir oyun/bölüm bitti.
            if "episode" in info:
                # Oyun bittiği için bölüm sayacını 1 artırıyoruz.
                self.episode_count += 1
                # Monitor tarafından hesaplanan, bu bölümdeki toplam ödülü 'r' (reward) anahtarından alıyoruz.
                score = info["episode"]["r"]
                # Bu skoru son 100 skoru tuttuğumuz listeye ekliyoruz.
                self.scores_window.append(score)
                
                # Listedeki son 100 oyunun ortalama skorunu numpy ile hesaplıyoruz.
                avg_score = np.mean(self.scores_window)
                
                # ERKEN DURDURMA KONTROLÜ: 700. bölüme gelindiyse ve ortalama skor hala 50'nin altındaysa
                if self.episode_count == 3000 and avg_score < 50.0:
                    print(f"\n[Erken Durdurma] 3000 bölüme ulaşıldı ama ortalama skor çok düşük ({avg_score:.2f} < 50). Parametreler elendi.")
                    self.is_pruned = True
                    return False
                # Modelin o anki rastgele hamle yapma (keşif) oranını (epsilon) alıyoruz.
                eps = self.model.exploration_rate
                
                # MLflow'a bu bölümdeki skoru, kaçıncı adım (bölüm) olduğu bilgisiyle kaydediyoruz.
                mlflow.log_metric("Score", score, step=self.episode_count)
                # MLflow'a 100 bölümlük hareketli ortalamayı kaydediyoruz.
                mlflow.log_metric("Avg_Score_100", avg_score, step=self.episode_count)
                # MLflow'a güncel epsilon değerini kaydediyoruz.
                mlflow.log_metric("Epsilon", eps, step=self.episode_count)
                
                # Konsol çok dolmasın diye sadece her 10 bölümde bir VEYA hedef skora ulaşılmışsa ekrana yazdırıyoruz.
                if self.episode_count % 10 == 0 or avg_score >= 195.0:
                    # Ekrana bölüm numarası, ortalama skor ve epsilon değerlerini basıyoruz.
                    print(f"Bölüm: {self.episode_count}\tSon 100 Bölüm Ortalama Skor: {avg_score:.2f}\tEpsilon: {eps:.2f}")
                
                # CartPole-v1 oyununun "çözülmüş" sayılması için son 100 oyunda ortalama 195 puan alması gerekir. Kontrol ediyoruz.
                if avg_score >= 195.0:
                    # Hedefe ulaşıldıysa tebrik mesajı yazdırıyoruz.
                    print(f"\nTEBRİKLER! Oyun Çözüldü! Ajan {self.episode_count}. bölümde başarıya ulaşıp denge kurmayı öğrendi.")
                    # False döndürmek, Stable Baselines 3'e "eğitimi burada durdur" emri vermektir.
                    return False
                    
        # True döndürmek "her şey yolunda, eğitime devam et" demektir.
        return True

# Optuna'nın deneyeceği hedef (objective) fonksiyonu tanımlıyoruz. Her trial (deneme) için baştan çalışacak.
def objective(trial):
    # MLflow'da her bir deneme için yeni bir alt (nested) run başlatıyoruz. İsim olarak Trial numarasını veriyoruz.
    with mlflow.start_run(run_name=f"Trial-{trial.number}", nested=True):
        
        # Öğrenme hızı (learning rate) için logaritmik olarak 0.0001 ile 0.01 arasında bir değer seç.
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        # Tek seferde hafızadan çekilecek veri sayısı (batch size) için bu 4 seçenekten birini seç.
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        # Gelecekteki ödüllerin önem katsayısı (gamma) için bu seçeneklerden birini seç.
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.99])
        # Hedef ağı güncelleme katsayısı (tau) için 0.005 ile 0.05 arasında bir değer seç.
        tau = trial.suggest_float("tau", 0.005, 0.05)
        # Keşif oranının (epsilon) ne kadar sürede azalacağını belirleyen fraction oranını seç (0.1 = eğitimin %10'u)
        exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.5)
        
        # Sabit hiperparametreler: Ajanın maksimum atacağı adım sayısı.
        total_timesteps = 250000 
        # Sabit hiperparametreler: Hafızada (replay buffer) tutulacak maksimum deneyim sayısı.
        buffer_size = 10000
        # Sabit hiperparametreler: Keşif oranının (epsilon) düşeceği en son (en küçük) değer.
        exploration_final_eps = 0.01
        
        # Optuna'nın bu deneme için seçtiği tüm hiperparametreleri MLflow'a kaydediyoruz.
        mlflow.log_params({
            "learning_rate": lr,
            "batch_size": batch_size,
            "gamma": gamma,
            "tau": tau,
            "exploration_fraction": exploration_fraction,
            "buffer_size": buffer_size,
            "total_timesteps": total_timesteps
        })
        
        # MLflow'da bu çalışmaya etiketler (tag) ekliyoruz, böylece sonradan filtrelemek kolaylaşır.
        mlflow.set_tag("framework", "stable-baselines3")
        mlflow.set_tag("optuna_trial", trial.number)
        
        # CartPole-v1 ortamını gym kullanarak oluşturuyoruz.
        env = gym.make("CartPole-v1")
        # Skor ve bölüm uzunluklarını takip edebilmek için ortamı Monitor ile sarmalıyoruz.
        env = Monitor(env)
        
        # Stable Baselines 3'ün DQN algoritmasını başlatıyoruz.
        model = DQN(
            "MlpPolicy", # Ajanın beyni olarak Standart Sinir Ağı (Multi Layer Perceptron) kullanılacak.
            env, # Oynanacak oyun ortamı
            learning_rate=lr, # Seçilen öğrenme hızı
            buffer_size=buffer_size, # Hafıza kapasitesi
            learning_starts=batch_size, # Öğrenmeye başlamadan önce hafızada en az batch_size kadar veri birikmesini bekle
            batch_size=batch_size, # Hafızadan bir seferde çekilecek anı sayısı
            tau=tau, # Hedef ağı yumuşak (soft) güncelleme oranı
            gamma=gamma, # İskonto faktörü (gelecekteki ödüllerin önemi)
            train_freq=4, # Ajan her 4 adım attığında 1 kere öğrenme yapacak
            gradient_steps=1, # Her öğrenme sırası geldiğinde sadece 1 kez ağırlıkları güncelleyecek
            target_update_interval=1, # Hedef ağ tau ile harmanlanarak her adımda (1) yavaş yavaş güncellenecek
            exploration_fraction=exploration_fraction, # Keşif oranının azalma süresi
            exploration_final_eps=exploration_final_eps, # Epsilon'un en son varacağı dip değer
            policy_kwargs=dict(net_arch=[64, 64]), # Sinir ağının yapısı: 2 tane gizli katman, her birinde 64 nöron var
            verbose=0, # Modelin kendi içine gömülü olan yazdırma işlemini kapatıyoruz (Konsolu biz yöneteceğiz)
            device="auto" # GPU varsa GPU'da, yoksa CPU'da eğitim yapılmasını sağlıyoruz
        )
        
        # Ekrana denemenin başladığını haber veriyoruz.
        print(f"\n--- Trial {trial.number} Başlıyor ---")
        
        # Yazdığımız kendi Callback sınıfımızdan bir kopya (nesne) oluşturuyoruz.
        callback = MLflowCallback()
        # Modeli, belirlediğimiz adım sayısı ve callback fonksiyonu ile eğitime başlatıyoruz.
        model.learn(total_timesteps=total_timesteps, callback=callback)
        # Eğitim bitince, bu deneme için açılmış olan simülasyon ortamını kapatıyoruz.
        env.close()
        
        # Eğer eğitim erken durdurulduysa (pruned), bu denemeyi başarısız olarak işaretleyip bir sonrakine geç.
        if callback.is_pruned:
            mlflow.log_metric("trial_result_score", 2000) # Başarısızlık cezası
            raise optuna.exceptions.TrialPruned()
            
        # Deneme bitti, şimdi hedefe ulaşılıp ulaşılmadığına bakıyoruz. Hiç skor yoksa 0 varsay.
        avg_score = np.mean(callback.scores_window) if len(callback.scores_window) > 0 else 0
        
        # Eğer ajan 195 hedefine ulaşmayı başardıysa:
        if avg_score >= 195.0:
            # Kaçıncı bölümde bu hedefe ulaştığını (episode_count) skor olarak belirliyoruz.
            # (Optuna bu skoru küçültmeye çalışacak, yani en hızlı ulaşanı bulacak)
            result_score = callback.episode_count
            
            # Bu başarıya ulaşan modeli bir zip dosyası olarak trial numarasıyla kaydediyoruz.
            model_path = f'cartpole_dqn_model_trial_{trial.number}.zip'
            model.save(model_path)
            # Kaydettiğimiz bu dosyayı MLflow'un içine bir Artifact olarak yüklüyoruz.
            mlflow.log_artifact(model_path)
            # Kaydettiğimizi ekrana yazdırıyoruz.
            print(f"Model kaydedildi: {model_path}")
            
        else:
            # Eğer verilen süre (200,000 adım) bittiyse ve ajan hala 195 ortalamaya ulaşamadıysa,
            # Optuna'nın bu parametrelerin "kötü" olduğunu anlaması için çok yüksek (kötü) bir ceza puanı döndürüyoruz.
            result_score = 2000 
            
        # Optuna'nın bulduğu bu nihai skoru da MLflow'a kaydediyoruz.
        mlflow.log_metric("trial_result_score", result_score)
        
        # Optuna'ya "Bu parametrelerle aldığımız skor budur" diyerek sonucu döndürüyoruz.
        return result_score


# Eğer bu dosya doğrudan çalıştırılmışsa (başka bir yerden import edilmemişse) aşağıdaki kodlar çalışır.
if __name__ == "__main__":
    # MLflow'un log dosyalarını yerel klasörde saklamasına izin veriyoruz.
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    # Logların nereye (mlruns klasörüne) kaydedileceğini ayarlıyoruz.
    mlflow.set_tracking_uri("file:./mlruns")
    # Tüm bu denemelerin (study) MLflow içinde hangi deney (experiment) başlığı altında toplanacağını belirliyoruz.
    mlflow.set_experiment("CartPole_DQN_Optuna_Tuning")
    
    # Tüm denemeleri kapsayan, hepsinin "babası" (parent) niteliğinde ana bir run başlatıyoruz.
    with mlflow.start_run(run_name="Optuna_Optimization"):
        
        print("Optuna Optimizasyon Süreci Başlıyor (20 Deneme Yapılacak)...")
        
        # Optuna'da bir 'Study' (çalışma) başlatıyoruz. direction="minimize" diyerek amacımızın 
        # result_score'u (harcanan bölüm sayısını) en aza indirmek olduğunu söylüyoruz.
        study = optuna.create_study(direction="minimize")
        # n_trials=20 diyerek, tanımladığımız objective fonksiyonunu 20 kere farklı rastgele parametrelerle çalıştırıyoruz.
        study.optimize(objective, n_trials=20)
        
        # Optimizasyon süreci bittiğinde en iyi sonuçları ekrana basıyoruz.
        print("\n=== OPTİMİZASYON TAMAMLANDI ===")
        # En başarılı denemenin numarasını yazdırıyoruz.
        print("En iyi Trial: ", study.best_trial.number)
        # O denemede elde edilen en iyi skoru (kaç bölümde hedefe ulaşıldığını) yazdırıyoruz.
        print("En iyi Sonuç (Bölüm Sayısı): ", study.best_value)
        # Bu başarıyı getiren hiperparametre kombinasyonunu yazdırıyoruz.
        print("En İyi Parametreler: ", study.best_trial.params)
        
        # Bulunan bu en iyi hiperparametreleri "best_..." öneki ile MLflow'daki ana run'ın içine kaydediyoruz.
        mlflow.log_params({"best_" + k: v for k, v in study.best_trial.params.items()})
        # En iyi skoru da metrik olarak kaydediyoruz.
        mlflow.log_metric("best_trial_score", study.best_value)

        # En iyi olan modelin dosya ismini belirliyoruz.
        best_model_name = f"cartpole_dqn_model_trial_{study.best_trial.number}"
        # Kullanıcıya test için bu dosyanın ismini nereye yazacağını söylüyoruz.
        print(f"\nEn iyi model '{best_model_name}.zip' olarak kaydedilmiştir.")
        print(f"Bu modeli test etmek için test_model.py içinde yükleme kısmını şu şekilde güncelleyebilirsiniz:")
        print(f"model = DQN.load(\"{best_model_name}\")")