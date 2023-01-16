import collections
import re
import random
import torch
def read_time_machine():
    with open('C:/Users/Chenglin Li/Desktop/data.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    #print(lines)
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines] #使用正则表达式对文本内容进行化简

lines = read_time_machine()
print(f'lines of the txt: {len(lines)}')
print(lines)

def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

tokens = tokenize(lines) #tokens是一个list of lists
for i in range(11):
    print(tokens[i])

def count_corpus(tokens):
    """统计词元的频率"""
    #tokens可能是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list): #判定是否是一个2D tokens
        # 如果是2D tokens，则将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line] #相当于一个循环嵌套，先执行for line in tokens
                                                              #再执行for token in line
    return collections.Counter(tokens)

class Vocab:
    '''将词映射到从零开始的数字索引'''
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x : x[1], reverse=True)
        # 未知词元的索引为0
        self.unk = 0
        self.idx_to_token = ['<unk>'] + reserved_tokens #用来装所有的词元，其在列表中的位置就是索引号
        self.token_to_idx = {token: idx #用来装所有的词元和其索引号，键是词元，值是对应索引号
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx: #如果这个token还未创建索引
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens): #已知token返回它的索引号
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def token_freqs(self):
        return self._token_freqs

vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab.__getitem__(tokens[i]))

#-----读取长序列数据-----#
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    #减1是因为从0位置开始分区和从num_step位置开始分区的到的子序列是一样的
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签,如果不减1则最后一组子序列没有标签可用
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        #返回样本X和标签Y
        #对于语言建模，目标是基于到目前为止我们看到的词元来预测下一个词元， 因此标签是移位了一个词元的原始序列。
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]  #由于循环网络多了时间特征，所以一个样本的标签也具有时间特征，
                                                              #因此每一个subseq是长度为num_step的一句话，则标签也是一句话（序列）
        yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    #最后所使用的文本长度
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1) #这样可以保证相邻batch中*对应位置*的子序列相邻
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y