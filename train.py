import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.utils.serialization import load_lua
import models
import data.dataloader as dataloader
import data.utils as utils
import data.dict as dict
from optims import Optim
from collections import defaultdict
import tensorflow as tf 



import os
import shutil
import argparse
import time
import math
import collections
from datetime import datetime
import codecs
from rouge import FilesRouge
from rouge import Rouge 
rouge = Rouge()



#config
parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('-config', default='default.yaml', type=str,
                    help="config file")
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-restore', default='', type=str,
                    help="restore checkpoint")
parser.add_argument('-seed', type=int, default=1234,
                    help="Random seed")
parser.add_argument('-model', default='seq2seq', type=str,
                    help="Model selection")
parser.add_argument('-score', default='', type=str,
                    help="score_fn")
parser.add_argument('-log', default='', type=str,
                    help="log directory")
parser.add_argument('-unk', action='store_true',
                    help="replace unk")
parser.add_argument('-reduce', action='store_true',
                    help="reduce redundancy")
parser.add_argument('-loss', default='', type=str,
                    help="loss function")
parser.add_argument('-weight', type=float, default=0.3,
                    help="weight")
parser.add_argument('-update', type=int, default=0,
                    help="pretrain updates")

opt = parser.parse_args()
config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)

#checkpoint
if opt.restore:
    print('loading checkpoint...\n')
    checkpoints = torch.load(opt.restore)
    config = checkpoints['config']


start_epoch = 1
if opt.restore:
    start_epoch = checkpoints['epoch']
    print("start epoch {}".format(start_epoch))


#cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus)>0
if use_cuda:
    torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)

#data
train_tag = 0
print('loading data...\n')
start_time = time.time()
trainset = torch.load(config.train)['train']
datas = torch.load(config.data)
print('loading time cost: %.3f' % (time.time()-start_time))

validset, testset = datas['valid'], datas['test']
src_vocab, tgt_vocab = datas['dicts']['src'], datas['dicts']['tgt']

if not (hasattr(config, 'src_vocab') or hasattr(config, 'tgt_vocab')):
    config.src_vocab = src_vocab.size()
    config.tgt_vocab = tgt_vocab.size()

if hasattr(config, 'eval_batch_size'):
    eval_batch_size = config.eval_batch_size
else:
    eval_batch_size = config.batch_size

padding = dataloader.padding
trainloader = dataloader.get_loader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=0, padding=padding)
validloader = dataloader.get_loader(validset, batch_size=eval_batch_size, shuffle=False, num_workers=0, padding=padding)
testloader = dataloader.get_loader(testset, batch_size=eval_batch_size, shuffle=False, num_workers=0, padding=padding)

#model
print('building model...\n')
model = getattr(models, opt.model)(config, config.src_vocab, config.tgt_vocab, use_cuda,
                                   score_fn=opt.score, weight=opt.weight, pretrain_updates=opt.update,
                                   extend_vocab_size=tgt_vocab.size()-config.tgt_vocab, device_ids=opt.gpus)


if opt.restore:
    model.load_state_dict(checkpoints['model'])
if use_cuda:
    model.cuda()


#optimizer
if opt.restore:
    optim = checkpoints['optim']
else:
    optim = Optim(config.optim, config.learning_rate, config.max_grad_norm, lr_decay=config.learning_rate_decay,start_decay_at=config.start_decay_at)
optim.set_parameters(model.parameters())

param_count = 0
for param in model.parameters():
    param_count += param.view(-1).size()[0]

#log
if not os.path.exists(config.log):
    os.mkdir(config.log)
if opt.log == '':
    log_path = config.log + str(datetime.now()).split('.')[0].split()[0] + '/'
else:
    log_path = config.log + opt.log + '/'

if os.path.exists(log_path):
    shutil.rmtree(log_path)
os.mkdir(log_path)

logging = utils.logging(log_path+'log.txt')
for k, v in config.items():
    logging("%s:\t%s\n" % (str(k), str(v)))
logging("\n")
logging(repr(model)+"\n\n")

logging('total number of parameters: %d\n' % param_count)
logging('score function is %s\n' % opt.score)


def record(file):
    def write_record(s):
        with open(file, 'a') as f:
            f.write(s)
    return write_record
recording = record(log_path + "record.txt")


event_path = log_path + "event" + "/"
if not os.path.exists(event_path):
    os.makedirs(event_path)
summary_writer = tf.summary.FileWriter(event_path)
tf_sum = tf.Summary()


#checkpoint
if opt.restore:
    updates = checkpoints['updates']
else:
    updates = 0
total_loss, start_time = 0, time.time()
report_total, report_correct = 0, 0
scores = [[] for metric in config.metric]
scores = collections.OrderedDict(zip(config.metric, scores))
best_scores = defaultdict(int)
if opt.restore:
    best_scores = checkpoints['best_scores']


#train
def train(epoch):
    model.train()
    global updates, total_loss, start_time, report_correct, report_total, trainloader
    print('train data size: %d ' % trainloader.__len__())
    for batch in trainloader:
        model.zero_grad()
        src, src_len, key, key_len, tgt, tgt_len = batch['src'], batch['src_len'], batch['key'], batch['key_len'], batch['tgt'], batch['tgt_len']

        num_oovs = 0
        loss, num_total, num_correct = model.train_model(src, src_len, key, key_len, tgt, tgt_len, opt.loss, updates, optim, num_oovs=num_oovs)

        total_loss += loss.item()
        report_correct += num_correct.item()
        report_total += num_total.item()

        utils.progress_bar(updates, config.eval_interval)
        updates += 1

        if updates % 3000 == 0:
            recording("train " + str(epoch)+" " + str(updates)+" " + str(total_loss/report_total) + "\n")

        if updates % config.eval_interval == 0:
            logging("epoch: %2d, ppl: %6.8f, time: %6.3f, updates: %3d, accuracy: %2.2f\n"
                    % (epoch, total_loss/report_total, time.time()-start_time, updates, report_correct*100.0 / report_total))
                
            print('evaluating after %d updates...\r' % updates)
            score = eval(epoch)
            for metric in config.metric:
                scores[metric].append(score[metric])

                if score[metric] >= max(scores[metric]):
                    save_model(log_path+'best_'+metric+'_checkpoint.pt')
            
            model.train()
            total_loss, start_time = 0, time.time()
            report_correct, report_total = 0, 0

        if updates % config.save_interval == 0:
            save_model(log_path+'checkpoint.pt')

    logging('\n')
    optim.updateLearningRate(score=0, epoch=epoch)
    logging('\n')


#evaluate
def eval(epoch, dataset=validloader, total_count=validset.__len__(), flag="valid"):
    model.eval()
    reference, candidate, source, alignments = [], [], [], []
    count = 0

    for batch in dataset:
        src, src_len, key, key_len, tgt, tgt_len = batch['src'], batch['src_len'], batch['key'], batch['key_len'], batch['tgt'], batch['tgt_len']
        
        num_oovs = 0
        oovs = None

        if config.beam_size == 1:
            samples, alignment = model.sample(src, src_len, key, key_len, num_oovs=num_oovs)
        else:
            samples, alignment = model.beam_sample(src, src_len, key, key_len, beam_size=config.beam_size)

        if oovs is not None:
            candidate += [tgt_vocab.convertToLabels(s.tolist(), dict.EOS, oovs=oov) for s, oov in zip(samples, oovs)]
        else:
            candidate += [tgt_vocab.convertToStrIds(s.tolist(), dict.EOS) for s in samples]
        
        source += [src_vocab.convertToStrIds(s.tolist(), dict.PAD) for s in src.t()]
        reference += [tgt_vocab.convertToStrIds(t.tolist(), dict.EOS) for t in tgt[1:].t()]
        alignments += [align.tolist() for align in alignment]

        count += src.size(1)
        print('\r', str(count)+'/'+str(total_count), end="", flush=True)
    print('\r', end='')
    print('total count: %d'.ljust(20) % total_count)

    if opt.unk:
        cands = []
        for s, c, align in zip(source, candidate, alignments):
            cand = []
            for word, idx in zip(c, align):
                if word == str(dict.UNK)+" " and idx < len(s):
                    try:
                        cand.append(s[idx])
                    except:
                        cand.append(word)
                        print("%d %d\n" % (len(s), idx))
                else:
                    cand.append(word)

            cands.append(cand)
        candidate = cands

    score = {}
    
    result = utils.eval_rouge(reference, candidate, log_path)
    try:
        score['rouge-1'] = result['F_measure'][0]
        score['rouge-2'] = result['F_measure'][1]
        score['rouge-L'] = result['F_measure'][2]
        logging("F_measure: %s\n" % str(result["F_measure"]))
        logging("recall: %s\n" % str(result["recall"]))
        logging("precision: %s\n" % str(result["precision"]))
        logging("\n")
    except:
        logging("Failed to compute rouge score.\n")
        score['rouge-1'] = 0.0
        score['rouge-2'] = 0.0
        score['rouge-L'] = 0.0

    if flag=="test":
        collect_rouge("roueg-1", score['rouge-1'], epoch)
        collect_rouge("roueg-2", score['rouge-2'], epoch)
        collect_rouge("roueg-L", score['rouge-L'], epoch)
        recording("test "+str(epoch)+" "+"rouge-1:"+str(score['rouge-1'])+" "+"rouge-2:"+str(score['rouge-2'])+" "+"rouge-l:"+str(score['rouge-L'])+"\n")

    return score


def save_model(path):
    global updates, best_scores, start_epoch
    checkpoints = {
        'model': model.state_dict(),
        'config': config,
        'optim': optim,
        'updates': updates,
        'best_scores': best_scores,
        'epoch': start_epoch}
    torch.save(checkpoints, path)


def collect_rouge(rouge_tag, rouge_score, epoch):
    tf_sum.value.add(tag=rouge_tag, simple_value=rouge_score)
    summary_writer.add_summary(tf_sum, epoch)


def main():
    global train_tag, trainloader, start_epoch
    for i in range(start_epoch, config.epoch+1):
        start_epoch = i
        train(i)

        logging("test ...")
        score = eval(i, testloader, testset.__len__(), "test")
        for metric in config.metric:
            best_scores[metric] = max(best_scores[metric], score[metric])

        # data
        start_time = time.time()
        config.train = config.train.replace(str(train_tag), str(i%3))
        train_tag = i%3
        trainset = torch.load(config.train)['train']
        print('loading time cost: %.3f' % (time.time()-start_time))
        print(config.train)
        trainloader = dataloader.get_loader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=0, padding=padding)

        logging("\n")

        for k, v in best_scores.items():
            logging(k+" : "+str(v)+'\n')
        logging('\n')

if __name__ == '__main__':
    main()
