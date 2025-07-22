import os
import time
import torch
import torchvision
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from dataset import MyDataset
import argparse
from model import MMFMixer_mission

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    # 设置随机数种子
    setup_seed(100)
    # Create model
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    my_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = MyDataset(path=r'./data/features', shp_path=r'./data/new/Wuhan_label.shp',
                              images_path=r'G:\Bag-of-Visual-Words-master\Code Files\data\images',
                              transform=my_transform)
    val_dataset = MyDataset(path=r'./data/features', shp_path=r'./data/new/test.shp',
                            images_path=r'G:\Bag-of-Visual-Words-master\Code Files\data\images', transform=my_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    print("Train numbers:{:d}".format(len(train_dataset)))
    print("Val numbers:{:d}".format(len(val_dataset)))

    model2 = MMFMixer_mission(args.num_class)

    print('model2 parameters:', sum(p.numel() for p in model2.parameters() if p.requires_grad))
    # print('model3 parameters:', sum(p.numel() for p in model3.parameters() if p.requires_grad))

    # model1 = model1.to(device)
    model2 = model2.to(device)
    # model3 = model3.to(device)
    # cost1 = nn.CrossEntropyLoss().to(device)
    cost2 = nn.CrossEntropyLoss().to(device)
    # cost2 = CB_loss(0.9999, 2.0).to(device)
    # cost3 = nn.CrossEntropyLoss().to(device)
    # Optimization
    # optimizer1 = optim.Adam(model1.parameters(), lr=args.lr, weight_decay=1e-6)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr, weight_decay=1e-6)
    # optimizer2 = torch.optim.SGD(model2.parameters(), lr=args.lr, momentum=0.9)
    # scheduler2 = torch.optim.lr_scheduler.OneCycleLR(optimizer2,max_lr=0.9,total_steps=100, verbose=False)
    # scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min', factor=0.9, patience=4, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0.000001, eps=1e-08)
    # criterion=nn.CrossEntropyLoss()
    # optimizer3 = optim.Adam(model3.parameters(), lr=args.lr, weight_decay=1e-6)

    # best_acc_1 = 0.
    best_acc_2 = 0.
    best_epoch = 0
    # best_acc_3 = 0.

    for epoch in range(1, args.epochs + 1):
        # model1.train()
        model2.train()
        # model3.train()
        # start time
        start = time.time()
        index = 0
        for doc_e, graph_e, poi_l, poi_g, rs_f, sv_f, labels in train_loader:
            # print(images.shape)
            labels = labels.to(device)
            doc_e = doc_e.to(device)
            graph_e = graph_e.to(device)
            poi_l = poi_l.to(device)
            rs_f = rs_f.to(device)
            sv_f = sv_f.to(device)

            # checkin = checkin.to(device)
            # checkin = checkin.clone().detach().float()

            # Forward pass
            # outputs1 = model1(images)
            outputs2 = model2(doc_e, graph_e, poi_l, rs_f, sv_f)
            # outputs2 = model2(rs_f)
            # outputs3 = model3(images)
            # loss1 = cost1(outputs1, labels)
            loss2 = cost2(outputs2, labels)
            # loss3 = cost3(outputs3, labels)

            # if index % 10 == 0:
            # print (loss)
            # Backward and optimize
            # optimizer1.zero_grad()
            optimizer2.zero_grad()
            # optimizer3.zero_grad()
            # loss1.backward()
            loss2.backward()
            # loss3.backward()
            # optimizer1.step()
            optimizer2.step()
            # scheduler2.step(loss2)
            # optimizer3.step()
            index += 1

        if epoch % 1 == 0:
            end = time.time()
            # print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, loss1.item(), (end-start) * 2))
            print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, loss2.item(), (end - start) * 2))
            # print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, loss3.item(), (end-start) * 2))

            # model1.eval()
            model2.eval()
            # model3.eval()

            # classes = ('bareland', 'cropland', 'forest', 'impervious', 'shrub', 'water')
            classes = ("居住区", "商业区", "工业区", "公共服务区")
            # classes = ('Airport', 'BareLand', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert', 'Farmland', 'Forest', 'Industrial', 'Meadow', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Pond', 'Port', 'RailwayStation', 'Resort', 'River', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct')
            # classes = ('1 industrial land', '10 shrub land', '11 natural grassland', '12 artificial grassland', '13 river', '14 lake', '15 pond', '2 urban residential', '3 rural residential', '4 traffic land', '5 paddy field', '6 irrigated land', '7 dry cropland', '8 garden plot', '9 arbor woodland')
            class_correct1 = list(0. for i in range(args.num_class))
            class_total1 = list(0. for i in range(args.num_class))
            class_correct2 = list(0. for i in range(args.num_class))
            class_total2 = list(0. for i in range(args.num_class))
            class_correct3 = list(0. for i in range(args.num_class))
            class_total3 = list(0. for i in range(args.num_class))
            class_correct_all = list(0. for i in range(args.num_class))
            class_total_all = list(0. for i in range(args.num_class))
            correct_prediction_1 = 0.
            total_1 = 0
            correct_prediction_2 = 0.
            total_2 = 0
            correct_prediction_3 = 0.
            total_3 = 0
            correct_prediction_all = 0.
            total_all = 0
            with torch.no_grad():
                for doc_e, graph_e, poi_l, poi_g, rs_f, sv_f, labels in val_loader:
                    labels = labels.to(device)
                    doc_e = doc_e.to(device)
                    graph_e = graph_e.to(device)
                    poi_l = poi_l.to(device)
                    rs_f = rs_f.to(device)
                    sv_f = sv_f.to(device)

                    # checkin = checkin.to(device)
                    # checkin = checkin.clone().detach().float()

                    # Forward pass
                    # outputs1 = model1(images)
                    outputs2 = model2(doc_e, graph_e, poi_l, rs_f, sv_f)
                    # outputs2 = model2(rs_f)

                    _2, predicted2 = torch.max(outputs2, 1)
                    c2 = (predicted2 == labels).squeeze()
                    # print(len(labels))
                    for label_idx in range(len(labels)):
                        # print(label_idx)
                        label = labels[label_idx]
                        # print(label)
                        class_correct2[label] += c2[label_idx].item()
                        class_total2[label] += 1
                    total_2 += labels.size(0)
                    # add correct
                    correct_prediction_2 += (predicted2 == labels).sum().item()

            for i in range(args.num_class):
                print('Model ResNeXt - Accuracy of %5s : %2d %%: Correct Num: %d in Total Num: %d' % (
                    classes[i], 100 * class_correct2[i] / class_total2[i], class_correct2[i], class_total2[i]))
            acc_2 = correct_prediction_2 / total_2
            print("Total Acc Model: %.4f" % (correct_prediction_2 / total_2))
            print('----------------------------------------------------')

        if acc_2 > best_acc_2:
            print('save new best acc_2', acc_2)
            torch.save(model2, os.path.join(args.model_path, 'MMF-without-3.pth'))
            best_acc_2 = acc_2
            best_epoch = epoch
        # if acc_3 > best_acc_3:
        #     print('save new best acc_3', acc_3)
        #     torch.save(model3, os.path.join(args.model_path, 'AID-30-teacher-densenet121-%s.pth' % (args.model_name)))
        #     best_acc_3 = acc_3
    # print("Model save to %s."%(os.path.join(args.model_path, 'UFZ-teacher-model-%s.pth' % (args.model_name))))
    # print('save new best acc_1', best_acc_1)
    print('save new best acc_2', best_acc_2, best_epoch)
    # print('save new best acc_3', best_acc_3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train hyper-parameter')
    parser.add_argument("--num_class", default=4, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    # parser.add_argument("--net", default='ResNet50', type=str)
    # parser.add_argument("--depth", default=50, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    # parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--model_name", default='', type=str)
    parser.add_argument("--model_path", default='./model_20250209', type=str)
    parser.add_argument("--pretrained", default=False, type=bool)
    parser.add_argument("--pretrained_model", default='', type=str)
    args = parser.parse_args()

    main(args)