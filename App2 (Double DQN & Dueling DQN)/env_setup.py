from pettingzoo.butterfly import cooperative_pong_v6
import supersuit as ss

def create_env():
    # 1. Ortamı oluşturuyoruz.
    # Atari tabanlı Pong Windows üzerinde C++ derleme hatası verdiği için,
    # PettingZoo'nun Butterfly kütüphanesindeki "Cooperative Pong" oyununu kullanıyoruz.
    # Bu ortamda iki raket var ve topu düşürmemeye çalışıyorlar. 
    # Öğrenme mantığı (MARL, D3QN, RLlib) tamamen aynıdır.
    env = cooperative_pong_v6.env()
    
    # 2. SuperSuit ile Preprocessing (Ön İşleme)
    # Renkleri gri tonlamaya çevirip boyutu küçültüyoruz.
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v1(env, x_size=84, y_size=84)
    
    # - Peş peşe gelen 4 kareyi birleştiririz (Frame Stacking)
    # Topun hareketini anlaması için şart.
    env = ss.frame_stack_v1(env, 4)
    
    return env

if __name__ == "__main__":
    env = create_env()
    env.reset()
    print("Ortamdaki Ajanlar (Oyuncular):", env.agents)
    print("Olası Aksiyonlar (paddle_0):", env.action_space(env.agents[0]))
    print("Gözlem (Observation) Boyutu (paddle_0):", env.observation_space(env.agents[0]).shape)
    print("Ortam başarıyla kuruldu ve çalışıyor!")
    env.close()
