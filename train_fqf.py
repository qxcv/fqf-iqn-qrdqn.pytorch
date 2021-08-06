import os
import yaml
import argparse
from datetime import datetime

from fqf_iqn_qrdqn.env import make_pytorch_env
from fqf_iqn_qrdqn.agent import FQFAgent


def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = make_pytorch_env(args.env_id, episode_life=args.atari,
                           clip_rewards=args.atari, atari=args.atari)
    obs_shape = env.observation_space.shape
    # make sure we have channels first, frame stack
    assert len(obs_shape) == 3, obs_shape
    assert obs_shape[0] <= min(obs_shape[1:]), obs_shape
    assert obs_shape[1] == obs_shape[2], obs_shape
    test_env = make_pytorch_env(
        args.env_id, episode_life=False, clip_rewards=False,
        atari=args.atari)

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{name}-seed{args.seed}-{time}')

    # Create the agent and run.
    agent = FQFAgent(
        env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
        cuda=args.cuda, **config)
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'fqf.yaml'))
    parser.add_argument('--env_id', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--no-atari', default=True, action='store_false',
                        dest='atari', help='use a non-Atari env')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    run(args)
