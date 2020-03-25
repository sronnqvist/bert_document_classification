from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertConfig, BertModel
from pytorch_transformers.modeling_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_transformers.tokenization_bert import BertTokenizer
#from tokenizers import BertWordPieceTokenizer as BertTokenizer
from torch import nn
import torch,math,logging,os
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


from .document_bert_architectures import DocumentBertLSTM, DocumentBertLinear, DocumentBertTransformer, DocumentBertMaxPool

def encode_documents(documents: list, tokenizer: BertTokenizer, max_input_length=512, max_sequences=16, name='unknown'):
    """
    Returns a len(documents) * max_sequences_per_document * 3 * 512 tensor where len(documents) is the batch
    dimension and the others encode bert input.

    This is the input to any of the document bert architectures.

    :param documents: a list of text documents
    :param tokenizer: the sentence piece bert tokenizer
    :return:
    """
    #tokenized_documents = [tokenizer.tokenize(document) for document in documents]
    #tokenized_documents = [tokenizer.encode(document) for document in documents]
    max_sequences_per_document = max_sequences #math.ceil(max(len(x)/(max_input_length-2) for x in tokenized_documents))
    #assert max_sequences_per_document <= 20, "Your document is to large, arbitrary size when writing"

    output = torch.zeros(size=(len(documents), max_sequences_per_document, 3, 512), dtype=torch.long)
    document_seq_lengths = [] #number of sequence generated per document
    #Need to use 510 to account for 2 padding tokens
    tokenized_documents = []
    for doc_index, document in enumerate(documents):
        if doc_index % 100 == 0:
            logging.info("Tokenizing %d" % doc_index)
        tokenized_document = tokenizer.tokenize(document)

        max_seq_index = 0
        for seq_index, i in enumerate(range(0, len(tokenized_document), (max_input_length-2))):
            if seq_index >= max_sequences_per_document:
                logging.info("Reached max seq count for doc with %s tokens." % len(tokenized_document))
                break
            raw_tokens = tokenized_document[i:i+(max_input_length-2)]
            #raw_tokens = tokenized_document.ids[i:i+(max_input_length-2)]
            tokens = []
            input_type_ids = []

            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in raw_tokens:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_masks = [1] * len(input_ids)

            while len(input_ids) < max_input_length:
                input_ids.append(0)
                input_type_ids.append(0)
                attention_masks.append(0)

            assert len(input_ids) == 512 and len(attention_masks) == 512 and len(input_type_ids) == 512

            #we are ready to rumble
            output[doc_index][seq_index] = torch.cat((torch.LongTensor(input_ids).unsqueeze(0),
                                                           torch.LongTensor(input_type_ids).unsqueeze(0),
                                                           torch.LongTensor(attention_masks).unsqueeze(0)),
                                                          dim=0)
            max_seq_index = seq_index
        document_seq_lengths.append(max_seq_index+1)
    torch.save(output, "%s_doc_encodings.pt" % name)
    torch.save(torch.LongTensor(document_seq_lengths), "%s_doc_lengths.pt" % name)
    return output, torch.LongTensor(document_seq_lengths)








document_bert_architectures = {
    'DocumentBertLSTM': DocumentBertLSTM,
    'DocumentBertTransformer': DocumentBertTransformer,
    'DocumentBertLinear': DocumentBertLinear,
    'DocumentBertMaxPool': DocumentBertMaxPool
}

class BertForDocumentClassification():
    def __init__(self,args=None,
                 labels=None,
                 device='cuda',
                 bert_model_path='bert-base-uncased',
                 architecture="DocumentBertLSTM",
                 batch_size=10,
                 bert_batch_size=7,
                 learning_rate = 5e-5,
                 weight_decay=0,
                 use_tensorboard=False):
        if args is not None:
            self.args = vars(args)
        if not args:
            self.args = {}
            self.args['bert_model_path'] = bert_model_path
            self.args['device'] = device
            self.args['learning_rate'] = learning_rate
            self.args['weight_decay'] = weight_decay
            self.args['batch_size'] = batch_size
            self.args['labels'] = labels
            self.args['bert_batch_size'] = bert_batch_size
            self.args['architecture'] = architecture
            self.args['use_tensorboard'] = use_tensorboard
        if 'fold' not in self.args:
            self.args['fold'] = 0

        assert self.args['labels'] is not None, "Must specify all labels in prediction"

        self.log = logging.getLogger()
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.args['bert_model_path'])
        #self.bert_tokenizer = BertTokenizer("/home/samuel/code/regbert/bert-base-uncased-vocab.txt", lowercase=True, )
        self.dev_document_representations = None
        self.dev_document_sequence_lengths = None
        self.dev_correct_output = None


        #account for some random tensorflow naming scheme
        if os.path.exists(self.args['bert_model_path']):
            if os.path.exists(os.path.join(self.args['bert_model_path'], CONFIG_NAME)):
                config = BertConfig.from_json_file(os.path.join(self.args['bert_model_path'], CONFIG_NAME))
            elif os.path.exists(os.path.join(self.args['bert_model_path'], 'bert_config.json')):
                config = BertConfig.from_json_file(os.path.join(self.args['bert_model_path'], 'bert_config.json'))
            else:
                raise ValueError("Cannot find a configuration for the BERT based model you are attempting to load.")
        else:
            config = BertConfig.from_pretrained(self.args['bert_model_path'])
        config.__setattr__('num_labels',len(self.args['labels']))
        config.__setattr__('bert_batch_size',self.args['bert_batch_size'])

        if 'use_tensorboard' in self.args and self.args['use_tensorboard']:
            assert 'model_directory' in self.args is not None, "Must have a logging and checkpoint directory set."
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_writer = SummaryWriter(os.path.join(self.args['model_directory'],
                                                                 "..",
                                                                 "runs",
                                                                 self.args['model_directory'].split(os.path.sep)[-1]+'_'+self.args['architecture']+'_'+str(self.args['fold'])))


        self.bert_doc_classification = document_bert_architectures[self.args['architecture']].from_pretrained(self.args['bert_model_path'], config=config)
        self.optimizer = torch.optim.Adam(
            self.bert_doc_classification.parameters(),
            weight_decay=self.args['weight_decay'],
            lr=self.args['learning_rate']
        )



    def fit(self, train, dev):
        """
        A list of
        :param documents: a list of documents
        :param labels: a list of label vectors
        :return:
        """

        train_documents, train_labels = train
        dev_documents, dev_labels = dev

        self.bert_doc_classification.train()

        try:
            document_representations = torch.load("train_doc_encodings.pt")
            document_sequence_lengths = torch.load("train_doc_lengths.pt")
            self.log.info("Loaded pre-tokenized training data.")
        except FileNotFoundError:
            document_representations, document_sequence_lengths  = encode_documents(train_documents, self.bert_tokenizer, name="train", max_sequences=self.args['bert_batch_size'])

        correct_output = torch.FloatTensor(train_labels)

        loss_weight = ((correct_output.shape[0] / torch.sum(correct_output, dim=0))-1).to(device=self.args['device'])
        self.loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=loss_weight)

        assert document_representations.shape[0] == correct_output.shape[0]

        if torch.cuda.device_count() > 1:
            self.bert_doc_classification = torch.nn.DataParallel(self.bert_doc_classification)
        self.bert_doc_classification.to(device=self.args['device'])
        self.bert_doc_classification.cuda()

        for epoch in range(1,self.args['epochs']+1):
            self.log.info("Training epoch %d" % epoch)
            # shuffle
            permutation = torch.randperm(document_representations.shape[0])
            document_representations = document_representations[permutation]
            document_sequence_lengths = document_sequence_lengths[permutation]
            correct_output = correct_output[permutation]

            self.epoch = epoch
            epoch_loss = 0.0
            for i in range(0, document_representations.shape[0], self.args['batch_size']):
                if i % 100 == 0:
                    self.log.info("  batch %d" % i)
                batch_document_tensors = document_representations[i:i + self.args['batch_size']].to(device=self.args['device'])
                batch_document_sequence_lengths= document_sequence_lengths[i:i+self.args['batch_size']]
                #self.log.info(batch_document_tensors.shape)
                batch_predictions = self.bert_doc_classification(batch_document_tensors,
                                                                 batch_document_sequence_lengths,
                                                                 freeze_bert=self.args['freeze_bert'], device=self.args['device'])

                batch_correct_output = correct_output[i:i + self.args['batch_size']].to(device=self.args['device'])
                loss = self.loss_function(batch_predictions, batch_correct_output)
                epoch_loss += float(loss.item())
                #self.log.info(batch_predictions)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss /= int(document_representations.shape[0] / self.args['batch_size'])  # divide by number of batches per epoch

            if 'use_tensorboard' in self.args and self.args['use_tensorboard']:
                self.tensorboard_writer.add_scalar('Loss/Train', epoch_loss, self.epoch)

            self.log.info('Epoch %i Completed: %f' % (epoch, epoch_loss))

            if epoch % self.args['checkpoint_interval'] == 0:
                self.save_checkpoint(os.path.join(self.args['model_directory'], "checkpoint_%s" % epoch))

            # evaluate on development data
            if epoch % self.args['evaluation_interval'] == 0:
                self.predict((dev_documents, dev_labels))

    def predict(self, data, threshold=0):
        """
        A tuple containing
        :param data:
        :return:
        """
        self.log.info('Evaluating on Epoch %i' % (self.epoch))
        if self.dev_document_representations is None:
            if isinstance(data, list):
                try:
                    self.dev_document_representations = torch.load("dev_doc_encodings.pt")
                    self.dev_document_sequence_lengths = torch.load("dev_doc_lengths.pt")
                    self.log.info("Loaded pre-tokenized validation data.")
                except FileNotFoundError:
                    self.dev_document_representations, self.dev_document_sequence_lengths = encode_documents(data, self.bert_tokenizer, name="dev", max_sequences=self.args['bert_batch_size'])
            if isinstance(data, tuple) and len(data) == 2:
                try:
                    self.dev_document_representations = torch.load("dev_doc_encodings.pt")
                    self.dev_document_sequence_lengths = torch.load("dev_doc_lengths.pt")
                    self.log.info("Loaded pre-tokenized validation data.")
                except FileNotFoundError:
                    self.dev_document_representations, self.dev_document_sequence_lengths = encode_documents(data[0], self.bert_tokenizer, name="dev", max_sequences=self.args['bert_batch_size'])

                self.dev_correct_output = torch.FloatTensor(data[1]).transpose(0,1)
                assert self.args['labels'] is not None
            torch.save(self.dev_correct_output, os.path.join(self.args['model_directory'], "dev_gold.pt"))

        self.log.info('  %i documents' % len(self.dev_document_representations))

        self.bert_doc_classification.to(device=self.args['device'])
        self.bert_doc_classification.eval()
        with torch.no_grad():
            predictions = torch.empty((self.dev_document_representations.shape[0], len(self.args['labels'])))
            for i in range(0, self.dev_document_representations.shape[0], self.args['batch_size']):
                if i % self.args['batch_size']*100 == 0:
                    self.log.info("  doc %d" % i)
                batch_document_tensors = self.dev_document_representations[i:i + self.args['batch_size']].to(device=self.args['device'])
                batch_document_sequence_lengths= self.dev_document_sequence_lengths[i:i+self.args['batch_size']]

                prediction = self.bert_doc_classification(batch_document_tensors,
                                                          batch_document_sequence_lengths,device=self.args['device'])
                predictions[i:i + self.args['batch_size']] = prediction

        for r in range(0, predictions.shape[0]):
            for c in range(0, predictions.shape[1]):
                if predictions[r][c] > threshold:
                    predictions[r][c] = 1
                else:
                    predictions[r][c] = 0
        predictions = predictions.transpose(0, 1)

        self.bert_doc_classification.train()
        if self.dev_correct_output is None:
            return predictions.cpu()
        else:
            assert self.dev_correct_output.shape == predictions.shape
            precisions = []
            recalls = []
            fmeasures = []

            for label_idx in range(predictions.shape[0]):
                correct = self.dev_correct_output[label_idx].cpu().view(-1).numpy()
                predicted = predictions[label_idx].cpu().view(-1).numpy()
                present_f1_score = f1_score(correct, predicted, average='binary', pos_label=1)
                present_precision_score = precision_score(correct, predicted, average='binary', pos_label=1)
                present_recall_score = recall_score(correct, predicted, average='binary', pos_label=1)

                precisions.append(present_precision_score)
                recalls.append(present_recall_score)
                fmeasures.append(present_f1_score)
                #logging.info('F1\t%s\t%f' % (self.args['labels'][label_idx], present_f1_score))

            micro_f1 = f1_score(self.dev_correct_output.T.numpy(), predictions.T.numpy(), average='micro')
            macro_f1 = f1_score(self.dev_correct_output.T.numpy(), predictions.T.numpy(), average='macro')
            self.log.info('Micro F1\t%f' % (micro_f1))

            if 'use_tensorboard' in self.args and self.args['use_tensorboard']:
                for label_idx in range(predictions.shape[0]):
                    self.tensorboard_writer.add_scalar('Precision/%s/Test' % self.args['labels'][label_idx].replace(" ", "_"), precisions[label_idx], self.epoch)
                    self.tensorboard_writer.add_scalar('Recall/%s/Test' % self.args['labels'][label_idx].replace(" ", "_"), recalls[label_idx], self.epoch)
                    self.tensorboard_writer.add_scalar('F1/%s/Test' % self.args['labels'][label_idx].replace(" ", "_"), fmeasures[label_idx], self.epoch)
                self.tensorboard_writer.add_scalar('Micro-F1/Test', micro_f1, self.epoch)
                self.tensorboard_writer.add_scalar('Macro-F1/Test', macro_f1, self.epoch)

            with open(os.path.join(self.args['model_directory'], "eval_%s.csv" % self.epoch), 'w') as eval_results:
                eval_results.write('Metric\t' + '\t'.join([self.args['labels'][label_idx] for label_idx in range(predictions.shape[0])]) +'\n' )
                eval_results.write('Precision\t' + '\t'.join([str(precisions[label_idx]) for label_idx in range(predictions.shape[0])]) + '\n' )
                eval_results.write('Recall\t' + '\t'.join([str(recalls[label_idx]) for label_idx in range(predictions.shape[0])]) + '\n' )
                eval_results.write('F1\t' + '\t'.join([ str(fmeasures[label_idx]) for label_idx in range(predictions.shape[0])]) + '\n' )
                eval_results.write('Micro-F1\t' + str(micro_f1) + '\n' )
                eval_results.write('Macro-F1\t' + str(macro_f1) + '\n' )
            #np.savetxt(os.path.join(self.args['model_directory'], "predictions_%s.txt" % self.epoch), predictions.numpy(), delimiter='\t')
            torch.save(predictions, os.path.join(self.args['model_directory'], "predictions_%s.pt" % self.epoch))


    def save_checkpoint(self, checkpoint_path: str):
        """
        Saves an instance of the current model to the specified path.
        :return:
        """
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        else:
            raise ValueError("Attempting to save checkpoint to an existing directory")
        self.log.info("Saving checkpoint: %s" % checkpoint_path )

        #save finetune parameters
        net = self.bert_doc_classification
        if isinstance(self.bert_doc_classification, nn.DataParallel):
            net = self.bert_doc_classification.module
        torch.save(net.state_dict(), os.path.join(checkpoint_path, WEIGHTS_NAME))
        #save configurations
        net.config.to_json_file(os.path.join(checkpoint_path, CONFIG_NAME))
        #save exact vocabulary utilized
        self.bert_tokenizer.save_vocabulary(checkpoint_path)
