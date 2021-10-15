import matplotlib.pyplot as plt
import numpy as np
import argparse
from discord_webhook import DiscordWebhook, DiscordEmbed
import os
from dotenv import load_dotenv

load_dotenv(verbose=True)

url = os.getenv('URL')
print(url)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, required=True,
                    help='either "train" or "test"')
args = parser.parse_args()

a = np.load(f'linear_rl_trader_rewards/{args.mode}.npy')

print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")

if args.mode == 'train':
  # show the training progress
  plt.plot(a)
else:
  # test - show a histogram of rewards
  plt.hist(a, bins=20)

plt.title(args.mode)
plt.savefig('result.png', dpi=300)

webhook = DiscordWebhook(url=url)

embed = DiscordEmbed(title='Plot', description='Reward', color='03b2f8')

embed.add_embed_field(name='average reward', value=f'{a.mean():.2f}')
embed.add_embed_field(name='min', value=f'{a.min():.2f}')
embed.add_embed_field(name='max', value=f'{a.max():.2f}')


# plt.savefig('result.png', dpi=300)

with open('result.png', 'rb') as f:
  webhook.add_file(file=f.read(), filename='plot.png')
embed.set_thumbnail(url='attachment://plot.png')


webhook.add_embed(embed)
response = webhook.execute()

# plt.show()