"""
Unfortunately, one cannot include the exact data utilized to the train both the clinical models due to HIPPA constraints.
The data can be found here if you fill out the appropriate online forms:
https://portal.dbmi.hms.harvard.edu/data-challenges/

For training, simply alter the config.ini present in /examples file for your purposes. Relevant variables are:

model_storage_directory: directory to store logging information, tensorboard checkpoints, model checkpoints

bert_model_path: the file path to a pretrained bert mode. can be the pytorch-transformers alias.

labels: an ordered list of labels you are training against. this should match the order given in a .fit() instance.


"""

from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from bert_document_classification.document_bert import BertForDocumentClassification
from pprint import pformat

import time, logging, torch, configargparse, os, socket, collections, csv

log = logging.getLogger()

def _initialize_arguments(p: configargparse.ArgParser):
    p.add('--model_storage_directory', help='The directory caching all model runs')
    p.add('--bert_model_path', help='Model path to BERT')
    p.add('--labels', help='Numbers of labels to predict over', type=str)
    p.add('--architecture', help='Training architecture', type=str)
    p.add('--freeze_bert', help='Whether to freeze bert', type=bool)

    p.add('--batch_size', help='Batch size for training multi-label document classifier', type=int)
    p.add('--bert_batch_size', help='Batch size for feeding 510 token subsets of documents through BERT', type=int)
    p.add('--epochs', help='Epochs to train', type=int)
    #Optimizer arguments
    p.add('--learning_rate', help='Optimizer step size', type=float)
    p.add('--weight_decay', help='Adam regularization', type=float)

    p.add('--evaluation_interval', help='Evaluate model on test set every evaluation_interval epochs', type=int)
    p.add('--checkpoint_interval', help='Save a model checkpoint to disk every checkpoint_interval epochs', type=int)

    #Non-config arguments
    p.add('--cuda', action='store_true', help='Utilize GPU for training or prediction')
    p.add('--device')
    p.add('--timestamp', help='Run specific signature')
    p.add('--model_directory', help='The directory storing this model run, a sub-directory of model_storage_directory')
    p.add('--use_tensorboard', help='Use tensorboard logging', type=bool)
    args = p.parse_args()

    args.labels = [x for x in args.labels.split(', ')]





    #Set run specific envirorment configurations
    args.timestamp = time.strftime("run_%Y_%m_%d_%H_%M_%S") + "_{machine}".format(machine=socket.gethostname())
    args.model_directory = os.path.join(args.model_storage_directory, args.timestamp) #directory
    os.makedirs(args.model_directory, exist_ok=True)

    #Handle logging configurations
    log.handlers.clear()
    formatter = logging.Formatter('%(message)s')
    fh = logging.FileHandler(os.path.join(args.model_directory, "log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.setLevel(logging.INFO)
    log.addHandler(ch)
    log.info(p.format_values())


    #Set global GPU state
    if torch.cuda.is_available() and args.cuda:
        if torch.cuda.device_count() > 1:
            log.info("Using %i CUDA devices" % torch.cuda.device_count() )
        else:
            log.info("Using CUDA device:{0}".format(torch.cuda.current_device()))
        args.device = 'cuda'
    else:
        log.info("Not using CUDA :(")
        args.dev = 'cpu'

    return args


def read_tsv(filename, label_index=0, sentence_index=1):
    with open(filename) as tsv_file:
        for line in csv.reader(tsv_file, delimiter='\t'):
            try:
                yield (line[label_index], line[sentence_index])
            except:
                logging.info(line)
                raise


#DATA_PATH="/home/samuel/data/CORE-final"
DATA_PATH="../../multilabel_bert/neuro_classifier/multilabel/CORE-final"
csv.field_size_limit(1000000)


top_level_labels = set("NA OP IN ID HI IP LY SP OTHER".split())
def normalize_label(label):
    return ' '.join(sorted([l if l in top_level_labels else l.lower() for l in label.split()]))


if __name__ == "__main__":
    p = configargparse.ArgParser(default_config_files=["config.ini"])
    args = _initialize_arguments(p)

    torch.cuda.empty_cache()

    data = {}
    label_set = set()
    doc_lengths = []
    for dataset in ['train', 'dev', 'test']:
        logging.info("Preparing %s" % dataset)
        data[dataset] = [x for x in read_tsv(os.path.join(DATA_PATH, dataset+'.tsv'), sentence_index=1)]
        for label, doc in data[dataset]:
            doc_lengths.append(len(doc.split()))
            for part in label.split():
                label_set.add(part)

    """import numpy as np
    logging.info("Doc length percentiles:")
    for x in range(10, 105, 5):
        logging.info(x, np.percentile(doc_lengths, x))"""

    top_level_labels = set("NA OP IN ID HI IP LY SP OTHER".split())
    label_list = sorted([l if l in top_level_labels else l.lower() for l in label_set])
    #logging.info(', '.join(label_list))
    label2idx = {l:i for i,l in enumerate(label_list)}

    documents = collections.defaultdict(lambda: [])
    labels = collections.defaultdict(lambda: [])
    for dataset in ['train', 'dev', 'test']:
        for label, doc in data[dataset]:
            documents[dataset].append(' '.join(doc.split()[:8000]))
            target = [0]*len(label_list)
            for part in label.split():
                if part not in top_level_labels:
                    part = part.lower()
                target[label2idx[part]] = 1
            labels[dataset].append(target)

    """label_counter = collections.defaultdict(lambda: 0)
    for l,_ in data['dev']:
        for ll in l.split():
            label_counter[ll] += 1
    tot = sum(label_counter.values())
    for label in label_list:
        logging.info("%.8f\t%s" % (label_counter[label.upper()]/len(data['dev']), label))"""

    model = BertForDocumentClassification(args=args)

    #print("Data sizes:", len(documents['train']), len(documents['dev']), len(documents['test']))
    model.fit((documents['train'], labels['train']), (documents['dev'], labels['dev']))
