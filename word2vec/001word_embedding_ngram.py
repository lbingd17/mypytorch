# coding=UTF-8
#查看pytorch版本的方法
##pytorch-doc-zh/docs/1.0/nlp_word_embeddings_tutorial.md
import torch
print(torch.__version__)  #注意是双下划线

# 作者: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)
#-------------------------
#输出：

#tensor([[ 0.6614,  0.2669,  0.0617,  0.6213, -0.4519]],
       grad_fn=<EmbeddingBackward>)
#-------------------------
例子： N-Gram语言模型

回想一下，在n-gram语言模型中,给定一个单词序列向量，我们要计算的是是单词序列的第i个单词。 
在本例中，我们将在训练样例上计算损失函数，并且用反向传播算法更新参数。
#-------------------------

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# 我们用莎士比亚的十四行诗 Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
# 应该对输入变量进行标记，但暂时忽略。
# 创建一系列的元组，每个元组都是([ word_i-2, word_i-1 ], target word)的形式。
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
# 输出前3行，先看下是什么样子。
print(trigrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for context, target in trigrams:

        # 步骤 1\. 准备好进入模型的数据 (例如将单词转换成整数索引,并将其封装在变量中)
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # 步骤 2\. 回调torch累乘梯度
        # 在传入一个新实例之前，需要把旧实例的梯度置零。
        model.zero_grad()

        # 步骤 3\. 继续运行代码，得到单词的log概率值。
        log_probs = model(context_idxs)

        # 步骤 4\. 计算损失函数（再次注意，Torch需要将目标单词封装在变量里）。
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        # 步骤 5\. 反向传播更新梯度
        loss.backward()
        optimizer.step()

        # 通过调tensor.item()得到单个Python数值。
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)  # 用训练数据每次迭代，损失函数都会下降。

#-------------------------
输出：

[(['When', 'forty'], 'winters'), (['forty', 'winters'], 'shall'), (['winters', 'shall'], 'besiege')]
[518.6343855857849, 516.0739576816559, 513.5321269035339, 511.0085496902466, 508.5003893375397, 506.0077188014984, 503.52977323532104, 501.06553316116333, 498.6121823787689, 496.16915798187256]
 #-------------------------
