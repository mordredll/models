import torch
import torch.nn as nn
# from pytorch_pretrained import BertModel,BertTokenizer
from transformers import AlbertTokenizer, AlbertModel,BertModel,BertTokenizer

class Config(object):
    """
    配置参数
    """
    def __init__(self,dataset): #初始化就加载dataset
        self.model_name = 'virTohostBert'
        # 训练集
        self.train_path = dataset + '/data/val.txt'
        # 验证集
        self.dev_path = dataset + '/data/test.txt'
        # 测试集
        self.test_path = dataset + '/data/test.txt'
        # 类别 CNews/data/class.txt
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]
        # 模型训练结果的保存位置
        self.save_path = '/kaggle/working/save_dict/' + self.model_name + '.ckpt'
        # 配置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 若超过1000batch效果还没有提升，提前结束
        self.require_improvement = 1000

        # 类别数
        self.num_classes = len(self.class_list)
        # epoch 数
        self.num_epochs = 3
        # batch_size 每个批次的数目
        self.batch_size = 128
        # 每句话的处理长度（短填，长切）
        self.pad_size = 32
        # 学习率
        self.learning_rate = 1e-5
        # bert 预训练模型位置
        self.bert_path = '/kaggle/working/virTohost_class/bert-base-uncased'
        # bert的切词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # bert 隐层层个数
        self.hidden_size = 768

class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        # BertModel.from_pretrained()加载预训练模型
        self.bert = BertModel.from_pretrained(config.bert_path)
        # 直接用参数-false，根据下游任务微调参数-true
        for para in self.parameters():
            para.requires_grad = True
        # 全连接层 输入和bert层对齐，输出取决于分类个数
        self.fc = nn.Linear(config.hidden_size,config.num_classes)

    def forward(self, x):
        # x [ids-LongTensor, seq_len, mask]
        context = x[0]  # 对应输入的句子 shape[batch_size x pad_size : 128, 32]
        mask = x[2]  # 负责挖空，对padding部分进行mask shape[128,32]
        # output_all_encoded_layers=False 不需要所有层的输出，不需要encoded_layers的输出
        _, pooled = self.bert(context, attention_mask=mask, return_dict=False)  # shape[128x768]
        out = self.fc(pooled)  # shape[128,10]
        return out