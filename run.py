# coding: UTF-8
import argparse
from train_eval import train, init_network
from importlib import import_module  # 动态导入

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', default='TextCNN', type=str,
                    help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # init embedding, 搜狗新闻: embedding_SougouNews.npz, 腾讯: embedding_Tencent.npz, 随机初始化: random
    embedding = 'random' if args.embedding == 'random' else 'embedding_SougouNews.npz'
    model_name = args.model  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, init_seed, init_logger

    init_seed(num=1)  # 固定随机种子，保证每次结果相同

    # load data
    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    c = config
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)  # [(words_idx, label, seq_len),...]
    train_iter = build_iterator(train_data, config)  # 迭代器
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)  # 模型初始化
    if model_name != 'Transformer':
        init_network(model)  # 权重初始化
    train(config, model, train_iter, dev_iter, test_iter)
