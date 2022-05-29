import argparse
import os
import json
from sys import platlibdir
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', '-d', type=str)
parser.add_argument('--model_type', '-t', type=str)
args = parser.parse_args()



def gather_bert_data(model_dir, model_type, name="", eval_type='accuracy'):
    

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
                df['Samples'] = f"{name}_{dir[index:dir.find('_', index)]}"
            else: 
                df['N'] = '0'
                df['Samples'] = f"{name}_{dir[index:]}"

            data_list.append(df)

    return pd.concat(data_list)


def gather_nn_data(model_dir, model_type, name="", eval_type='accuracy'):
    
    data_list = []

    for dir in os.listdir(model_dir):
        if model_type in dir:
            data = pd.read_csv(f'{model_dir}/{dir}/siam_{name}_100dglove_test_metrics.csv')[['accuracy', 'epoch']]
            data = data.rename(columns={"accuracy": "Accuracy", "epoch":"Epochs"})
            index = dir.find('_', dir.find('_')+1)+1
            if 'augmented' in dir:
                data['N'] = dir[-1]
                data['Samples'] = f"{name}_{dir[index:dir.find('_', index)]}"
            else: 
                data['N'] = '0'
                data['Samples'] = f"{name}_{dir[index:]}"

            data_list.append(data)

    return pd.concat(data_list)


def plot_acc(data, model_type): 
    
    data.N = data.N.astype(int)
    data.Epochs = data.Epochs.astype(float)

    grid = sns.FacetGrid(data, col="N", hue="Samples", col_wrap=2, legend_out=True)
    bp = grid.map(sns.lineplot, 'Epochs', 'Accuracy')
    bp.set_titles("N={col_name}")
    bp.set_ylabels("Accuracy")
    bp.set_xlabels("Epochs")
    grid.add_legend(title="Models")
    grid.savefig(f"{model_type}.pdf")


if __name__ == '__main__':
    bert_data1 = gather_bert_data(args.model_dir, "bert_siamese", name="siam")
    bert_data2 = gather_bert_data(args.model_dir, "bert_classifier", name="class")
    data = pd.concat([bert_data1, bert_data2])
    plot_acc(data=data, model_type=args.model_type)
    
    nn_data1 = gather_nn_data(args.model_dir, "siamese_lstm", name="lstm")
    nn_data2 = gather_nn_data(args.model_dir, "siamese_lscnntm", name="lstmcnn")
    nn_data3 = gather_nn_data(args.model_dir, "siamese_cnn", name="cnn")
    data = pd.concat([nn_data1, nn_data2, nn_data3])
    plot_acc(data=data, model_type=args.model_type)