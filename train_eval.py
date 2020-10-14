# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif, get_logger
from torch.utils.tensorboard import SummaryWriter


def init_network(model, method='xavier', exclude='embedding', seed=123):
    # 权重初始化，默认xavier
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    # init logger and writer
    time_stamp = time.strftime('%m-%d_%H.%M', time.localtime())
    writer = SummaryWriter(log_dir=config.log_path + '/' + time_stamp)
    logger = get_logger(log_path=config.log_path + '/' + time_stamp + '/' + time_stamp + '_logging.txt')
    logger.warning(model.parameters)
    for key, value in config.__dict__.items():
        logger.warning('{}: {}'.format(key, value))

    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_step = 0                   # 记录进行到多少step
    dev_best_loss = float('inf')     # 正无穷，float('-inf')负无穷
    last_improve = 0                 # 记录上次验证集loss下降的step数
    flag = False                     # 记录是否很久没有效果提升
    for epoch in range(config.num_epochs):
        logger.warning('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)  # trains: (words_idx, seq_len)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_step % 100 == 0:
                labels = labels.data.cpu()  # cuda tensor转到cpu后才能转成numpy, 用于metrics计算
                predict = torch.max(outputs.data, 1)[1].cpu()  # 1-D
                train_acc = metrics.accuracy_score(labels, predict)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    state = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict()}
                    torch.save(state, config.save_path)  # 保存模型
                    improve = '*'
                    last_improve = total_step
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Step: {0:>4},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                logger.warning(msg.format(total_step, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_step)
                writer.add_scalar("loss/dev", dev_loss, total_step)
                writer.add_scalar("acc/train", train_acc, total_step)
                writer.add_scalar("acc/dev", dev_acc, total_step)
                model.train()
            total_step += 1
            if total_step - last_improve > config.require_improvement:  # 验证集loss超过1000个step没下降，结束训练
                logger.warning("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, model, test_iter, logger)


def test(config, model, test_iter, logger):
    # test
    model.load_state_dict(torch.load(config.save_path)['model_state_dict'])
    model.eval()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    logger.warning(msg.format(test_loss, test_acc))
    logger.warning("Precision, Recall and F1-Score...")
    logger.warning(test_report)
    logger.warning("Confusion Matrix...")
    logger.warning(test_confusion)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)