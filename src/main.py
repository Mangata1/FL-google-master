import copy
import numpy as np
import pickle
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
from models import  CNNMnist, CNNCifar
from utils import average_weights, args_info,get_dataset
from args import args_parser
from update import LocalUpdate, test_inference

#article:communication-Efficient Learning of Deep Networks from Decentralized data

#reference:https://github.com/AshwinRJ/Federated-Learning-PyTorch


if __name__ == '__main__':
    device ='cpu'
    args = args_parser()
    train_dataset, test_dataset, user_groups = get_dataset(args)
    args_info(args)

    # choose model
    if args.model == 'mlp':
        img_size = train_dataset[0][0].shape
        len_in = 1
        
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    elif args.model == 'cnn':

        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)    
    else:
        exit('model choose invalid !')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)



    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print('\n | Global Training Round :  ',epoch+1)
        '''
        def train(self, mode=True):
        r"""Sets the module in training mode."""      
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self
        '''
        global_model.train()#use  BatchNormalization and Dropout
		
		#FedAvg core 
		
        m = max(int(0.1 * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,idxs=user_groups[idx])
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,idxs=user_groups[idx])
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        if (epoch+1) % print_every == 0:
            print(' \nAvg Training Stats after {epoch+1} global rounds:')
            print('Training Loss : ',np.mean(np.array(train_loss)))
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))


    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(" Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print(" Test Accuracy: {:.2f}%".format(100*test_acc))


    file_name = '../save/{}_{}_{}_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs,  args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

