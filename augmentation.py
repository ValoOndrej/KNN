# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

from modules.data import *
from modules.utils import *

#arguments to be parsed from command line
import argparse

path = Path('./logs/data/')
if not (path/'dataset.csv').exists():
    get_quora_huggingface(path)

data_path = Path('./logs/data')

ap = argparse.ArgumentParser()
ap.add_argument("-in","--input", required=False, type=str, help="input file of unaugmented data", default=data_path/"dataset.csv")
ap.add_argument("-out","--output", required=False, type=str, help="output file of unaugmented data", default=data_path/"augmented_dataset.csv")
ap.add_argument("-num_aug", "--num_aug", required=False, type=int, help="number of augmented sentences per original sentence", default=9)
ap.add_argument("-sr","--alpha_sr", required=False, type=float, help="percent of words in each sentence to be replaced by synonyms", default=0.1)
ap.add_argument("-ri","--alpha_ri", required=False, type=float, help="percent of words in each sentence to be inserted", default=0.1)
ap.add_argument("-rs","--alpha_rs", required=False, type=float, help="percent of words in each sentence to be swapped", default=0.1)
ap.add_argument("-rd","--alpha_rd", required=False, type=float, help="percent of words in each sentence to be deleted", default=0.1)
args = ap.parse_args()

if args.alpha_sr == args.alpha_ri == args.alpha_rs == args.alpha_rd == 0:
     ap.error('At least one alpha should be greater than zero')

#main function
if __name__ == "__main__":

    data = ImportData(str(args.input))

    data.train_test_split(seed=44, augment=True)
    

