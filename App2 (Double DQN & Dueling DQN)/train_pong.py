import os
import ray
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env

# 1. Ortamımızı projeye dahil ediyoruz
from env_setup import create_env

def env_creator(args):
    # RLlib, PettingZoo ortamlarını doğrudan anlayamaz.
    # Onu RLlib'in anlayacağı bir "PettingZooEnv" sarmalayıcısına (wrapper) sokmamız gerekiyor.
    env = create_env()
    return PettingZooEnv(env)

if __name__ == "__main__":
    # Ray'i başlat
    ray.init()

    # Ortamı RLlib'e "coop_pong" ismiyle kaydediyoruz
    env_name = "coop_pong"
    register_env(env_name, env_creator)

    # Ortamın geçici bir kopyasını oluşturup ajanların (oyuncuların) isimlerini alıyoruz
    temp_env = env_creator({})
    agent_list = temp_env.possible_agents
    print("Ortamdaki Ajanlar:", agent_list)

    # 2. D3QN ve Multi-Agent Konfigürasyonu
    config = (
        DQNConfig()
        .environment(env=env_name)
        # --- D3QN AYARLARI ---
        .training(
            double_q=True,  # Double DQN aktif: Aşırı tahminleri (overestimation) önler
            dueling=True,   # Dueling DQN aktif: Durum (State) ve Aksiyon (Action) avantajını ayırır
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": 50000, # Deneyimlerin tutulacağı hafıza boyutu
            },
            num_steps_sampled_before_learning_starts=1000, # Öğrenmeye başlamadan önce rastgele oynayarak veri toplama
            train_batch_size=256,
        )
        # --- MULTI-AGENT AYARLARI ---
        .multi_agent(
            # Ortamdaki tüm ajanlar (paddle_0, paddle_1) aynı 'policy'yi (beyni) kullanacak.
            # Yani tek bir beyin eğiteceğiz ve iki raket de bu beyinden karar alacak (Parameter Sharing).
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
        )
        # --- SİSTEM AYARLARI ---
        .resources(num_gpus=1) # Cuda var demiştin, GPU'yu kullanması için 1 yapıyoruz
        .env_runners(num_env_runners=1) # Veri toplamak için kullanılacak CPU worker (işçi) sayısı
    )

    print("Konfigürasyon başarıyla oluşturuldu!")
    # Ray'i şimdilik kapatıyoruz, eğitimi bir sonraki adımda başlatacağız.
    ray.shutdown()
