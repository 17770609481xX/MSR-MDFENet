'''
coding:utf-8
@Software:PyCharm
@Time:2024/2/29 15:11
@Author:chenGuiBin
@description: 主运行文件
'''

import os
import argparse
import torch
import torch.optim as optim
import pandas as pd
# from model.MultiScaleFusionResidualNet import MSFRNet
from model.MultiScaleFusionResidualNet import MSFRNet
from util.utils import create_lr_scheduler, get_params_groups, train_one_epoch, evaluate
from DataLoader.GetDataloader import load_data

import warnings
warnings.filterwarnings("ignore")

def main(args):
    # gpu运行
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # 加载数据，采用留一法
    # train_data_loader, val_data_loader = load_data_train_test(args.data_path, args.train_name_list, args.val_name,
    #                                                                  args.batch_size)
    train_data_loader, val_data_loader = load_data(args.data_path, args.train_name_list, args.val_name,
                                                              args.batch_size)

    model = MSFRNet(in_channels=1, n_classes=args.num_classes).to(device)
    # model = ResNet18(in_channels=1, num_classes=args.num_classes).to(device)

    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs, warmup=True, warmup_epochs=1)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0.
    patience = 0
    train_result_ones = []
    test_result_ones = []
    result_one_last = []
    print('------------------LOSO validate name: {}  ------------------'.format(args.val_name))
    for epoch in range(args.epochs):
        # train
        train_loss, train_accuracy, train_precision, train_recall, train_f1_score, train_specificity, train_auc_roc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_data_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)
        # test
        test_loss, test_accuracy, test_precision, test_recall, test_f1_score, test_specificity, test_auc_roc = evaluate(model=model,
                                     data_loader=val_data_loader, device=device, epoch=epoch, name=args.val_name)

        train_result_ones.append([args.val_name, train_loss, train_accuracy, train_precision, train_recall, train_f1_score, train_specificity, train_auc_roc])
        test_result_ones.append([args.val_name, test_loss, test_accuracy, test_precision, test_recall, test_f1_score, test_specificity, test_auc_roc])

        result_one_last = [args.val_name, test_loss, test_accuracy, test_precision, test_recall, test_f1_score, test_specificity, test_auc_roc]

    # 直接将训练测试结果写到文档中
    train_result_ones_df = pd.DataFrame(train_result_ones, columns=['Name', 'loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'specificity', 'AUC'])
    test_result_ones_df = pd.DataFrame(test_result_ones,columns=['Name', 'loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'specificity', 'AUC'])
    # 写入Excel文件
    train_result_ones_df.to_excel('result/CHBMITTrainResultFile/train_' + args.val_name + '10.xlsx', index=False)
    test_result_ones_df.to_excel('result/CHBMITTrainResultFile/test_' + args.val_name + '10.xlsx', index=False)
    return result_one_last


def multi_person_run():
    name_list = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06', 'chb07', 'chb08', 'chb09', 'chb10',
                 'chb11', 'chb12', 'chb13', 'chb14', 'chb15', 'chb16', 'chb17', 'chb18', 'chb19', 'chb20',
                 'chb21', 'chb22', 'chb23']
    # name_list = ['ccy', 'fcx', 'hlj', 'jmy', 'ljx', 'lqs', 'scy', 'xyx', 'xzx', 'yyx',
    #              'zzh']
    # 保存结果写入excel中
    results = []
    results1 = []
    # 遍历数组，逐个测试
    for i in range(len(name_list)):
        current_element = name_list[i]
        # 获取当前元素前和当前元素后的所有元素组成新的数组
        new_arr = name_list[:i] + name_list[i + 1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_path', type=str, default="data/CHBMITDataSet.mat")# 0.5-45HZ，选择前额四导联
        # parser.add_argument('--data_path', type=str, default="data/JHMCHHDataSetFusion.mat")# 0.5-45HZ，选择前额四导联
        # parser.add_argument('--data_path', type=str, default="data/CHBMITDataSet11Channel.mat")# 0.5-45HZ，选择前额四导联
        # parser.add_argument('--data_path', type=str, default="data/CHBMITDataSet1.mat")# 0.5-45HZ，选择前额四导联
        parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--epochs', type=int, default=80)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--random_state', type=int, default=42)  # 这个参数干嘛用
        parser.add_argument('--test_size', type=int, default=0.1)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--wd', type=float, default=5e-4)
        parser.add_argument('--train_name_list', type=list, default=new_arr)
        parser.add_argument('--val_name', type=str, default=current_element)
        parser.add_argument('--channel', type=int, default=18)

        opt = parser.parse_args()
        result_one_last = main(opt)
        results.append(result_one_last)

    # 将结果写入excel中
    # 将数组转换为DataFrame
    result_df = pd.DataFrame(results, columns=['Name', 'loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'specificity', 'AUC'])
    # 写入Excel文件
    result_df.to_excel('result/CHBMITLOSO/CHBMITLOSO100.xlsx', index=False)


def one_person_run():
    train_name_list = ['chb01', 'chb02', 'chb03',  'chb05', 'chb06', 'chb07', 'chb19', 'chb09', 'chb10',
                 'chb11', 'chb12', 'chb04', 'chb14', 'chb15', 'chb16', 'chb17', 'chb18', 'chb13',
                 'chb20', 'chb21', 'chb22', 'chb23']
    val_name = 'chb08'# 轮流验证，验证随便指定，但是一定是22个人训练一个人测试，测试的这个人单独拿出来，也不能几个人一起测试，不好写准确率啥的，也没啥影响

    # 这里是23个人，name_lsit 就是训练的人
    # test_name就是验证的
    parser = argparse.ArgumentParser()

    # parser.add_argument('--data_path', type=str, default="data/DataSetKaggle.mat")
    parser.add_argument('--data_path', type=str, default="data/DataSet5S0545.mat")# 0.5-45HZ，无ICA
    # parser.add_argument('--data_path', type=str, default="data/DataSet5S0545FP.mat")  # 0.5-45HZ，选择前额四导联
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--EEGNet-name', default='RegNetY_400MF', help='create EEGNet name')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--random_state', type=int, default=42) #这个参数干嘛用
    parser.add_argument('--test_size', type=int, default=0.2)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--train_name_list', type=list, default=train_name_list)
    parser.add_argument('--val_name', type=str, default=val_name)
    parser.add_argument('--threshold', type=int, default=10)

    opt = parser.parse_args()
    result_one_best, result_one_last = main(opt)
    print(result_one_best)
    print(result_one_last)


if __name__ == '__main__':
    # one_person_run()  # 运行单个人的数据进行测试
    multi_person_run()


# 查看日志文件代码
# tensorboard --logdir=./SummaryWriterFile
# 888是两层，100是三层，10是叠加+两层网络