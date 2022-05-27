import argparse
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', '-d', type=str)
parser.add_argument('--model_type', '-t', type=str)
args = parser.parse_args()



def gather_data(model_dir, model_type, eval_type='accuracy'):
    

    data_list = []

    for dir in os.listdir(model_dir):
        if model_type in dir:
            with open(f'{model_dir}/{dir}/trainer_state.json', 'r') as f:
                data = json.load(f)
            metric = []
            epoch = []
            for record in data["log_history"]:
                if f'eval_{eval_type}' in record:
                    metric.append(record[f'eval_{eval_type}'])
                    epoch.append(record['epoch'])

            df = pd.DataFrame({'Accuracy':metric, 'Epochs':epoch})
            index = dir.find('_', dir.find('_')+1)+1
            if 'augmented' in dir:
                df['N'] = dir[-1]
                df['Samples'] = dir[index:dir.find('_', index)]
            else: 
                df['N'] = '0'
                df['Samples'] = dir[index:]

            data_list.append(df)

    return pd.concat(data_list)


def plot_acc(data, model_type): 
    
    data.N = data.N.astype(int)
    data.Epochs = data.Epochs.astype(float)
    data.Samples = data.Samples.astype(int)


    fig, axs = plt.subplots(nrows=2, ncols=2)
    ns = data.N.unique()

    for idx, ax in enumerate(axs.flat):
        df = data.loc[data['N'] == ns[idx]].copy().reset_index()
        sns.lineplot(data=df, x='Epochs', y='Accuracy',
                     hue='Samples', ax=ax, palette=sns.color_palette("hls", 3))
        ax.set_title(f'N={ns[idx]}')
        ax.set_ylabel('')
        ax.set_xlabel('')
        if idx==2:
            plt.legend(ncol=1, title='Samples')
        else:
            ax.get_legend().remove()

    fig.suptitle('Eval. accuracy throughout the training')
    fig.supxlabel('Epochs')
    fig.supylabel('Accuracy')
    fig.tight_layout()
    fig.savefig(f"{model_type}.pdf")


if __name__ == '__main__':
    data = gather_data(args.model_dir, args.model_type)
    plot_acc(data=data, model_type=args.model_type)

