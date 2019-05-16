import torch
import torch.nn as nn
from torch.autograd import Variable
import data.dict as dict
import models


class seq2seq(nn.Module):
    def __init__(self, config, src_vocab_size, tgt_vocab_size, use_cuda,
                 score_fn=None, weight=0.0, pretrain_updates=0, extend_vocab_size=0, device_ids=None):
        super(seq2seq, self).__init__()
        src_embedding = None
        tgt_embedding = None

        if 'copy' in score_fn:
            build_encoder = models.copy_rnn_encoder
            build_decoder = models.copy_rnn_decoder
        else:
            build_encoder = models.rnn_encoder
            build_decoder = models.rnn_decoder

        self.encoder = build_encoder(config, src_vocab_size, embedding=src_embedding)
        self.encoder_key = build_encoder(config, src_vocab_size, embedding=src_embedding)
        if config.shared_vocab == False:
            self.decoder = build_decoder(config, tgt_vocab_size, embedding=tgt_embedding, score_fn=score_fn, extend_vocab_size=extend_vocab_size)
        else:
            self.decoder = build_decoder(config, tgt_vocab_size, embedding=self.encoder.embedding, score_fn=score_fn, extend_vocab_size=extend_vocab_size)
                
        self.use_cuda = use_cuda
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.config = config
        self.weight = weight

        if 'copy' in score_fn:
            self.criterion = models.copy_criterion(use_cuda)
        else:
            self.criterion = models.criterion(tgt_vocab_size, use_cuda)
        
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def compute_loss(self, hidden_outputs, targets, loss_fn, updates):
        if loss_fn == 'copy':
            return models.copy_cross_entropy_loss(hidden_outputs, self.decoder, targets, self.criterion, self.config)
        else:
            return models.cross_entropy_loss(hidden_outputs, self.decoder, targets, self.criterion, self.config)

    def forward(self, src, src_len, key, key_len, tgt, tgt_len, num_oovs):
        lengths, indices = torch.sort(src_len.squeeze(0), dim=0, descending=True)
        src = torch.index_select(src, dim=1, index=indices)
        key = torch.index_select(key, dim=1, index=indices)
        tgt = torch.index_select(tgt, dim=1, index=indices)

        contexts, state = self.encoder(src, lengths.tolist())
        contexts_key, state_key = self.encoder_key(key, key_len.squeeze(0).tolist())
        outputs, final_state = self.decoder(tgt[:-1], state, contexts.transpose(0, 1), contexts_key.transpose(0, 1), src=src, num_oovs=num_oovs)
        
        return outputs, tgt[1:]

    def train_model(self, src, src_len, key, key_len, tgt, tgt_len, loss_fn, updates, optim, num_oovs=0):
        src = Variable(src)
        key = Variable(key)
        tgt = Variable(tgt)
        src_len = Variable(src_len).unsqueeze(0)
        key_len = Variable(key_len).unsqueeze(0)
        tgt_len = Variable(tgt_len).unsqueeze(0)
        if self.use_cuda:
            src = src.cuda()
            key = key.cuda()
            tgt = tgt.cuda()
            src_len = src_len.cuda()
            key_len = key_len.cuda()
            tgt_len = tgt_len.cuda()

        outputs, targets = self(src, src_len, key, key_len, tgt, tgt_len, num_oovs)
        loss, num_total, num_correct = self.compute_loss(outputs, targets, loss_fn, updates)
        loss.backward()
        optim.step()

        return loss, num_total, num_correct

    def sample(self, src, src_len, key, key_len, num_oovs=0):
        if self.use_cuda:
            src = src.cuda()
            key = key.cuda()
            src_len = src_len.cuda()
            key_len = key_len.cuda()

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, ind = torch.sort(indices)
        with torch.no_grad():
            src = Variable(torch.index_select(src, dim=1, index=indices))
            key = Variable(torch.index_select(key, dim=1, index=indices))
            bos = Variable(torch.ones(src.size(1)).long().fill_(dict.BOS))

        if self.use_cuda:
            bos = bos.cuda()

        contexts, state = self.encoder(src, lengths.tolist())
        contexts_key, state_key = self.encoder_key(key, key_len.tolist())
        sample_ids, final_outputs = self.decoder.sample([bos], state, contexts.transpose(0, 1), contexts_key.transpose(0, 1), src=src, num_oovs=num_oovs)
        _, attns = final_outputs
        alignments = attns.max(2)[1]

        sample_ids = torch.index_select(sample_ids, dim=1, index=ind)
        alignments = torch.index_select(alignments, dim=1, index=ind)

        return sample_ids.t(), alignments.t()


    def beam_sample(self, src, src_len, key, key_len, beam_size = 1, num_oovs=0, n_best = 1):
        batch_size = src.size(1)

        # (1) Run the encoder on the src. Done!!!!
        if self.use_cuda:
            src = src.cuda()
            key = key.cuda()
            src_len = src_len.cuda()
            key_len = key_len.cuda()

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, ind = torch.sort(indices)
        with torch.no_grad():
            src = Variable(torch.index_select(src, dim=1, index=indices))
            key = Variable(torch.index_select(key, dim=1, index=indices))
        contexts, encState = self.encoder(src, lengths.tolist())
        contexts_key, encState_key = self.encoder_key(key, key_len.tolist())

        with torch.no_grad():
            def var(a):
                return Variable(a)
        def rvar(a):
            return var(a.repeat(1, beam_size, 1))
        def bottle(m):
            return m.view(batch_size * beam_size, -1)
        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        contexts = rvar(contexts.data).transpose(0, 1)
        decState = (rvar(encState[0].data), rvar(encState[1].data))
        contexts_key = rvar(contexts_key.data).transpose(0, 1)

        beam = [models.Beam(beam_size, n_best=1, cuda=self.use_cuda) for _ in range(batch_size)]
        self.decoder.attention.init_context(contexts, contexts_key)

        summarys = []
        for i in range(self.config.max_tgt_len):
            if all((b.done() for b in beam)):
                break

            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.getCurrentState() for b in beam]).t().contiguous().view(-1))

            output, decState, attn, summarys = self.decoder.sample_one(inp, decState, summarys)

            # (b) Compute a vector of batch*beam word scores.
            output = unbottle(self.log_softmax(output))
            attn = unbottle(attn)
            last_hidden = unbottle(summarys[-1])

            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(output.data[:, j], attn.data[:, j])
                b.beam_update(decState, j)
                b.update(last_hidden, j)

            summarys[-1] = bottle(last_hidden)
            
        # (3) Package everything up.
        allHyps, allAttn = [], []
        for j in ind:
            b = beam[j]
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att.max(1)[1])
            allHyps.append(hyps[0])
            allAttn.append(attn[0])

        return allHyps, allAttn