'''
 @Author: Shuming Ma
 @mail:   shumingma@pku.edu.cn
 @homepage : shumingma.com
'''
import torch
import torch.nn as nn
import models
import data.dict as dict
from torch.autograd import Variable
from torch.nn.parameter import Parameter

def criterion(tgt_vocab_size, use_cuda):
    weight = torch.ones(tgt_vocab_size)
    weight[dict.PAD] = 0
    crit = nn.CrossEntropyLoss(weight, size_average=False)
    if use_cuda:
        crit.cuda()
    return crit

def nll_criterion(tgt_vocab_size, use_cuda):
    weight = torch.ones(tgt_vocab_size)
    weight[dict.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if use_cuda:
        crit.cuda()
    return crit

class copy_criterion(nn.Module):
    def __init__(self, use_cuda):
        super(copy_criterion, self).__init__()
        self.crit = nn.NLLLoss(size_average=False)
        self.use_cuda = use_cuda

    def forward(self, outputs, targets):
        batch_size, vocab_size = outputs.size()
        weight = torch.ones(vocab_size)
        weight[dict.PAD] = 0
        weight = Variable(weight.unsqueeze(0))
        if self.use_cuda:
            weight = weight.cuda()
        return self.crit(outputs*weight, targets)


def memory_efficiency_cross_entropy_loss(hidden_outputs, decoder, targets, criterion, config):
    outputs = Variable(hidden_outputs.data, requires_grad=True, volatile=False)
    num_total, num_correct, loss = 0, 0, 0

    outputs_split = torch.split(outputs, config.max_generator_batches)
    targets_split = torch.split(targets, config.max_generator_batches)
    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
        out_t = out_t.view(-1, out_t.size(2))
        scores_t = decoder.compute_score(out_t)
        loss_t = criterion(scores_t, targ_t.view(-1))
        pred_t = scores_t.max(1)[1]
        num_correct_t = pred_t.data.eq(targ_t.data).masked_select(targ_t.ne(dict.PAD).data).sum()
        num_total_t = targ_t.ne(dict.PAD).data.sum()
        num_correct += num_correct_t
        num_total += num_total_t
        loss += loss_t.data[0]
        loss_t.div(num_total_t).backward()

    grad_output = outputs.grad.data
    hidden_outputs.backward(grad_output)

    return loss, num_total, num_correct, config.tgt_vocab, config.tgt_vocab

def copy_cross_entropy_loss(hidden_outputs, decoder, targets, criterion, config, sim_score=0):
    loss, pred = [], []
    for id, output in enumerate(hidden_outputs):
        l = criterion(output, targets[id])

        if (l.data != l.data).any():
            print(output)
            print(targets[id])
            raise(ValueError)

        p = output.max(1)[1]
        loss.append(l)
        pred.append(p)
    loss = torch.sum(torch.stack(loss))
    pred = torch.stack(pred)

    num_correct = pred.data.eq(targets.data).masked_select(targets.ne(dict.PAD).data).sum()
    num_total = targets.ne(dict.PAD).data.sum()
    loss = loss / num_total.float()

    return loss, num_total, num_correct

def cross_entropy_loss(hidden_outputs, decoder, targets, criterion, config, sim_score=0, compute_score=True):
    outputs = hidden_outputs.view(-1, hidden_outputs.size(2))
    if compute_score:
        scores = decoder.compute_score(outputs)
    else:
        scores = outputs
    loss = criterion(scores, targets.view(-1)) + sim_score
    pred = scores.max(1)[1].view(targets.size())
    num_correct = pred.eq(targets).masked_select(targets.ne(dict.PAD).data).sum()
    num_total = targets.ne(dict.PAD).data.sum()
    loss = loss / num_total.float()

    return loss, num_total, num_correct
