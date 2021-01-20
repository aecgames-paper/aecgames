import numpy as np
import pandas as pd
import os
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
import sys

METHOD = sys.argv[1]

# FORMAT = "pgf"  # "png"
FORMAT = "pgf"
OUTPUT_FILENAME = "pursuitGraph"

if FORMAT == "pgf":
    matplotlib.use("pgf")
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.size": 9,
        "legend.fontsize": 9,
        "text.usetex": True,
        "pgf.rcfonts": False
    });
    # plt.figure(figsize=(2.65, 1.5))
    fig = plt.figure(figsize=(5, 2.8))
else:
    fig = plt.figure(figsize=(5, 2.8))

data_path = "./"
data_path_tweak = "./"
envs = ['unpruned', 'pruned']
scale_factor = 1
colors = [['#B7E4C7', '#74C69D', '#40916C', '#1B4332'],
          ['#ADE8F4', '#48CAE4', '#0096C7', '#023E8A']]

ax = fig.add_subplot(1,1,1)

for i in range(4):
    env_num = 0
    for env_name in envs:
        df = pd.read_csv(os.path.join(data_path, METHOD+'_'+env_name+'_'+str(i)+'.csv'))
        df = df[['episodes_total', "episode_reward_mean"]]
        data = df.to_numpy()
        window = int(len(data[:, 1])/100)
        filtered = signal.savgol_filter(data[:, 1],window+1 if window%2 == 0 else window,5)
        ax.plot(data[:, 0], filtered/scale_factor, label=env_name.capitalize()+' '+str(i), linewidth=0.75, linestyle='-', color=colors[env_num][i])
        env_num += 1

handles, labels = ax.get_legend_handles_labels()
handles = handles[0:len(envs)*4:2] + handles[1:len(envs)*4:2]
labels = labels[0:len(envs)*4:2] + labels[1:len(envs)*4:2]


plt.xlabel('Episode', labelpad=1)
plt.ylabel('Average Total Reward', labelpad=1)
plt.title('Pursuit')
plt.xticks(ticks=[10000,20000,30000,40000,50000],labels=['10k','20k','30k','40k','50k'])
plt.xlim(0, 60000)
if METHOD == 'ppo':
    plt.yticks(ticks=[60,70,80,90,100,110,120],labels=['60','70','80','90','100','110','120'])
    plt.ylim(50, 130)
    legend_loc = 'upper right'
elif METHOD == 'adqn':
    plt.yticks(ticks=[100,200,300,400,500,600],labels=['100','200','300','400','500','600'])
    plt.ylim(0, 700)
    legend_loc = 'lower left'
plt.tight_layout()
plt.legend(handles, labels, loc=legend_loc, ncol=2, labelspacing=.2, columnspacing=.25, borderpad=.25)
plt.margins(x=0)
plt.savefig("pruned_pursuit_"+METHOD+".pgf", bbox_inches = 'tight',pad_inches = .025)
plt.savefig("pruned_pursuit_"+METHOD+".png", bbox_inches = 'tight',pad_inches = .025, dpi = 600)

