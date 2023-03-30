import torch
import torch.nn as nn
from torchtext.vocab import GloVe
import torch.nn.functional as F
from torch.autograd import Variable


# class BiLSTM(nn.Module):
#     def __init__(self, vocab, embed_size=100, hidden_size=256, num_layers=2, dropout=0.1):
#         # num_layers=2表示有两个方向（双向），LSTM的参数bidirectional=True表示双向LSTM,默认为False
#         super(BiLSTM, self).__init__()
#         self.embedding = nn.Embedding(len(vocab), embed_size, padding_idx=vocab['<pad>'])
#         glove = GloVe(name="6B", dim=100)
#         self.embedding = nn.Embedding.from_pretrained(glove.get_vecs_by_tokens(vocab.get_itos()),
#                                                       padding_idx=vocab['<pad>'],
#                                                       freeze=True)
#
#         self.encoder = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, bidirectional=True, dropout=dropout)
#         self.decoder = nn.Linear(2 * hidden_size, 2)
#
#         # 初始时间步和最终时间步的隐藏状态作为全连接层输入
#         self.w_omega = nn.Parameter(torch.Tensor(
#             hidden_size * 2, hidden_size * 2))
#         print(self.w_omega.shape)
#         self.u_omega = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
#         print(self.u_omega.shape)
#
#         # 初始化函数（均匀分布，-0.1-0.1）
#         nn.init.uniform_(self.w_omega, -0.1, 0.1)
#         nn.init.uniform_(self.u_omega, -0.1, 0.1)
#
#     def forward(self, inputs):
#         # inputs的形状是(seq_len,batch_size)
#         embeddings = self.embedding(inputs)
#         print(embeddings.shape)
#         # 提取词特征，输出形状为(seq_len,batch_size,embedding_dim)
#         # rnn.LSTM只返回最后一层的隐藏层在各时间步的隐藏状态。
#         outputs, (h_n, _) = self.encoder(embeddings)  # output, (h, c)
#         print(outputs.shape)
#         # outputs形状是(seq_len,batch_size, 2 * num_hiddens)
#         # x = outputs.permute(1, 0, 2)
#         # print(x.shape)
#         # x形状是(batch_size, seq_len, 2 * num_hiddens)
#
#         # Attention过程
#         u = torch.tanh(torch.matmul(outputs, self.w_omega))
#         print(u.shape)
#         # u形状是(batch_size, seq_len, 2 * num_hiddens)
#         att = torch.matmul(u, self.u_omega)
#         print(att.shape)
#         # att形状是(batch_size, seq_len, 1)
#         att_score = F.softmax(att, dim=1)
#         print(att_score.shape)
#         # att_score形状仍为(batch_size, seq_len, 1)
#         scored_x = outputs * att_score
#         print(scored_x.shape)
#         # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
#         # Attention过程结束
#
#         feat = torch.sum(scored_x, dim=1)  # 加权求和
#         print(feat.shape)
#         # feat形状是(batch_size, 2 * num_hiddens)
#         outs = self.decoder(feat)
#         print(outs.shape)
#         # out形状是(batch_size, 2)
#         return outs

class BiLSTM(torch.nn.Module):
    def __init__(self, vocab, batch_size=128, output_size=2, hidden_size=256, embed_dim=300,
                 dropout=0.1, use_cuda=True, attention_size=256, sequence_length=512, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.use_cuda = use_cuda
        self.sequence_length = sequence_length
        self.lookup_table = nn.Embedding(len(vocab), embed_dim, padding_idx=vocab['<pad>'])
        glove = GloVe(name="42B", dim=300)
        self.lookup_table = nn.Embedding.from_pretrained(glove.get_vecs_by_tokens(vocab.get_itos()),
                                                         padding_idx=vocab['<pad>'],
                                                         freeze=True)
        #     self.lookup_table.weight.data.uniform_(-1., 1.)
        #
        #     self.layer_size = 2
        #     self.lstm = nn.LSTM(self.embed_dim,
        #                         self.hidden_size,
        #                         self.layer_size,
        #                         dropout=dropout,
        #                         bidirectional=True)
        #
        #     self.attention_size = attention_size
        #
        #     self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size).cuda())
        #     self.u_omega = Variable(torch.zeros(self.attention_size).cuda())
        #     self.label = nn.Linear(hidden_size * self.layer_size, output_size)
        #
        # # self.attn_fc_layer = nn.Linear()
        #
        # def attention_net(self, lstm_output):
        #     print(lstm_output)
        #     # print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)
        #     print(lstm_output.size())
        #     output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * self.layer_size])
        #     # print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)
        #     print(output_reshape.size())
        #     attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        #     # print(attn_tanh.size()) = (squence_length * batch_size, attention_size)
        #     print(attn_tanh.size())
        #     attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        #     # print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)
        #     # print(attn_hidden_layer.size())
        #     exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length])
        #     # print(exps.size()) = (batch_size, squence_length)
        #     # print(exps.size())
        #     alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        #     # print(alphas.size()) = (batch_size, squence_length)
        #     # print(alphas.size())
        #     alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_length, 1])
        #     # print(alphas_reshape.size()) = (batch_size, squence_length, 1)
        #     # print(alphas_reshape.size())
        #     state = lstm_output.permute(1, 0, 2)
        #     # print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)
        #     # print(state.size())
        #     attn_output = torch.sum(state * alphas_reshape, 1)
        #     # print(attn_output.size()) = (batch_size, hidden_size*layer_size)
        #     # print(attn_output.size())
        #     return attn_output
        #
        # def forward(self, input_sentences):
        #     input = self.lookup_table(input_sentences)
        #     input = input.permute(1, 0, 2)
        #
        #     h_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
        #     c_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
        #
        #     lstm_output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        #     attn_output = self.attention_net(lstm_output)
        #     logits = self.label(attn_output)
        #     return logits

        self.lookup_table.weight.data.uniform_(-1., 1.)

        self.layer_size = 1
        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_size,
                            self.layer_size,
                            dropout=dropout,
                            bidirectional=bidirectional)

        if bidirectional:
            self.layer_size = self.layer_size * 2
        else:
            self.layer_size = self.layer_size

        self.attention_size = attention_size
        if self.use_cuda:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size).cuda())
            self.u_omega = Variable(torch.zeros(self.attention_size).cuda())
        else:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size))
            self.u_omega = Variable(torch.zeros(self.attention_size))

        self.label = nn.Linear(hidden_size * self.layer_size, output_size)

    # self.attn_fc_layer = nn.Linear()

    def attention_net(self, lstm_output):
        # print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)

        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * self.layer_size])
        # print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        # print(attn_tanh.size()) = (squence_length * batch_size, attention_size)

        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        # print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length])
        # print(exps.size()) = (batch_size, squence_length)

        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        # print(alphas.size()) = (batch_size, squence_length)

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_length, 1])
        # print(alphas_reshape.size()) = (batch_size, squence_length, 1)

        state = lstm_output.permute(1, 0, 2)
        # print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)

        attn_output = torch.sum(state * alphas_reshape, 1)
        # print(attn_output.size()) = (batch_size, hidden_size*layer_size)

        return attn_output

    def forward(self, input_sentences, batch_size=None):
        input = self.lookup_table(input_sentences)
        input = input.permute(1, 0, 2)

        if self.use_cuda:
            h_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
        else:
            h_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))

        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        attn_output = self.attention_net(lstm_output)
        logits = self.label(attn_output)
        return logits
