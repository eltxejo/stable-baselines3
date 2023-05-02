from stable_baselines3 import PPO, SAC, DDPG

def learnPPO(policy, env, timesteps):
    # Train an agent
    model = PPO(policy, env, verbose=1, tensorboard_log="./_resultados/")
    # print(model.policy)
    model.learn(total_timesteps=timesteps) # tb_log_name="first_run"

    # return model

def learnSAC(policy, env, timesteps):
    # Train an agent
    model = SAC(policy, env, verbose=1, tensorboard_log="./_resultados/")
    # print(model.policy)
    model.learn(total_timesteps=timesteps) # tb_log_name="first_run"

    # return model

def runSB(model):
    # Run the model
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()


if __name__ == '__main__':
    policy = "MlpPolicy" # Define el modelo (red neuronal)
    env = "BipedalWalker-v3"
    timesteps = 10_000

    learnPPO(policy, env, timesteps)
    learnSAC(policy, env, timesteps)
    # runSB(model)