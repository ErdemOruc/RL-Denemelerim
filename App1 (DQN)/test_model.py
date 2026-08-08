import gymnasium as gym
from stable_baselines3 import DQN

print("Eğitilmiş model yükleniyor...")

# Ortamı insan gözüyle izleyebileceğimiz şekilde başlatıyoruz
env = gym.make("CartPole-v1", render_mode="human")

# Kaydettiğimiz zeki beyin ağırlıklarını (tecrübeyi) yüklüyoruz
# Stable Baselines 3, yapıyı (ağ boyutu vs.) otomatik olarak dosyanın içinden okur, 
# bu nedenle ekstra bir "boş beyin" oluşturmamıza gerek yoktur.
try:
    model = DQN.load("cartpole_dqn_model_sb3")
    print("Başarılı model yüklendi.")
except FileNotFoundError:
    print("Model bulunamadı! 'cartpole_dqn_model_sb3.zip' dosyasının olduğundan emin olun.")
    print("Eğitim hedefine tam ulaşılamadıysa, son kaydedilen model yükleniyor...")
    try:
        model = DQN.load("cartpole_dqn_model_son_hal_sb3")
        print("Son hal modeli yüklendi.")
    except FileNotFoundError:
        print("Hiçbir model dosyası bulunamadı. Lütfen önce DqnModel.py ile eğitimi çalıştırın.")
        env.close()
        exit()


print("Ajan ortama bırakıldı! Başarısını izleyin.")

for i_episode in range(1, 6): # Toplam 5 oyun izleyeceğiz
    state, info = env.reset()
    score = 0
    
    while True:
        # Ajan beyin gücünü kullanarak en iyi kararı veriyor (deterministic=True, yani hiç rastgelelik YOK!)
        # Bu fonksiyon manuel PyTorch tensor dönüşümlerinin yerini alır.
        action, _states = model.predict(state, deterministic=True)
        
        # Kararı uygula
        next_state, reward, done, truncated, info = env.step(action)
        
        state = next_state
        score += reward
        
        if done or truncated:
            print(f"Oyun {i_episode} Bitti! Skor: {score}")
            break

env.close()
