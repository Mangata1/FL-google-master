import copy
import torch
from torchvision import datasets, transforms
from sample import mnist_iid, mnist_noniid
from sample import cifar_iid, cifar_noniid

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def get_dataset(args):


    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose([transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#change image（0,255）->（0，1）and normlizated

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            
            user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' :
        data_dir = '../data/mnist/'
      
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])#均值是0.1307，标准差是0.3081，这些系数都是数据集提供方计算好的数据

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            
            user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups

def args_info(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('IID')
    else:
        print('Non-IID')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
