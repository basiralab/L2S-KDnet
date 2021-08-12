"""Main function of Learn to SuperResolve Brain Graphs with Knowledge Distillation Network (L2S-KDnet) framework 
   for predicting high-resolution brain connectomes from low-resolution connectomes based on teacher-student paradigm. 
    
    ---------------------------------------------------------------------
    
    This file contains the implementation of the training and testing process of our L2S-KDnet model.
        TS_model(train_lr_loader, train_hr_loader,  opts)
                TS_model:                       reference of TS model:    L2S-KDnet, Baseline etc.
                Inputs:
                        train_lr_loader:        low resolution train data loader for training: loader = get_loader(features, batch_size, train_test, num_workers=1) 
                                                    features:       (ns × nf) LR connectivity matrix
                                                        ns:             number of subjects
                                                        nf:             number of edges without self-loop in lr
                                                    batch_size:     size of lr features
                                                    train_test:     train or test checker for shuffle condition
                                                    num_workers:    number of workers that performs load operation    
                        train_hr_loader:        high resolution train data loader for training: loader = get_loader(features, batch_size, train_test, num_workers=1) 
                                                    features:       (ns × n'f) HR connectivity matrix
                                                        ns:             number of subjects
                                                        n'f:            number of edges without self-loop in hr
                                                    batch_size:     size of hr features
                                                    train_test:     train or test checker for shuffle condition
                                                    num_workers:    number of workers that performs load operation
                        opts:                   parsed command line arguments, to learn more about the arguments run: 
                                                    python demo.py --help
                Output:
                        returns constructed model            
        train(fold)
                Inputs:
                        fold:                   the number of current fold in cross-validation
                Output:     
                        returns nothing (void)
        test(test_lr_loader, test_hr_loader, fold)
                Inputs:
                        test_lr_loader:         low resolution train data loader for testing: loader = get_loader(features, batch_size, train_test, num_workers=1) 
                                                    features:       (ns × nf) LR connectivity matrix
                                                        ns:             number of subjects
                                                        nf:             number of edges without self-loop in lr
                                                    batch_size:     size of lr features
                                                    train_test:     train or test checker for shuffle condition
                                                    num_workers:    number of workers that performs load operation    
                        test_hr_loader:         high resolution train data loader for testing: loader = get_loader(features, batch_size, train_test, num_workers=1) 
                                                    features:       (ns × n'f) HR connectivity matrix
                                                        ns:             number of subjects
                                                        n'f:            number of edges without self-loop in hr
                                                    batch_size:     size of hr features
                                                    train_test:     train or test checker for shuffle condition
                                                    num_workers:    number of workers that performs load operation
                        fold:                   the number of current fold in cross-validation
                Outputs:
                        returns MAE, eigenvector and pageRank centrality testing errors for both teacher and student and returns KL divergence error for student
                        
    To evaluate our framework we used 3-fold cross-validation strategy.
    ---------------------------------------------------------------------
    Copyright 2021 Başar Demir, Istanbul Technical University.
    All rights reserved.
"""


import argparse

from sklearn.model_selection import KFold
from torch.backends import cudnn

from L2S_KDnet.prediction import *
from benchmark_methods.L2S_KDnet_wo_TD_regularization.prediction import *
from benchmark_methods.L2S_KDnet_wo_local_topology.prediction import *
from benchmark_methods.baseline.prediction import Baseline
from benchmark_methods.baseline_with_discriminator_for_decoder.prediction import *
from helpers import printFoldResults, printTestResults
import yaml

iteration = 150
parser = argparse.ArgumentParser()
# initialization
# Basic opts.
parser.add_argument('--model', type=str, default='L2S-KDnet')
parser.add_argument('--log_dir', type=str, default='logs/')
parser.add_argument('--checkpoint_dir', type=str, default='models/')
parser.add_argument('--sample_dir', type=str, default='samples/')
parser.add_argument('--result_dir', type=str, default='results/')
parser.add_argument('--plot_dir', type=str, default='plots/')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--result_root', type=str, default='output/')
parser.add_argument('--lr', type=str, default='simulated_data/morphological_data.npy',help='LR input data path')
parser.add_argument('--hr', type=str, default='simulated_data/functional_data.npy',help='HR input data path')
parser.add_argument('--num_splits', type=int, default=3)

parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')

# GCN model opts
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--in_feature_t0', type=int, default=595)
parser.add_argument('--hidden1', type=int, default=100)
parser.add_argument('--hidden2', type=int, default=50)
parser.add_argument('--hidden3', type=int, default=595)
parser.add_argument('--LRout', type=int, default=595)
parser.add_argument('--SRout', type=int, default=12720)

# model opts.
parser.add_argument('--t_lr', type=float, default=0.0001, help='learning rate for teacher')
parser.add_argument('--s_lr', type=float, default=0.0001, help='learning rate for student')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--local_topology_loss_weight', type=float, default=0.1, help='weight of the local topology loss')
parser.add_argument('--teacher_loss_weight', type=float, default=0.5, help='teacher loss weight')

# Training opts.
parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
parser.add_argument('--num_workers', type=int, default=0, help='num_workers to load data.')
parser.add_argument('--num_iters', type=int, default=iteration, help='number of total iterations for training')
parser.add_argument('--log_step', type=int, default=iteration)
parser.add_argument('--model_save_step', type=int, default=iteration)

# Test opts.
parser.add_argument('--test_iters', type=int, default=iteration, help='test model from this step')

# Path and folder operations
opts = parser.parse_args()
opts.result_root += (str(opts.model) + str("_" + str(opts.local_topology_loss_weight)) + str("_" + str(opts.teacher_loss_weight)))
opts.log_dir = os.path.join(opts.result_root, opts.log_dir)
opts.checkpoint_dir = os.path.join(opts.result_root, opts.checkpoint_dir)
opts.sample_dir = os.path.join(opts.result_root, opts.sample_dir)
opts.result_dir = os.path.join(opts.result_root, str(opts.result_dir))
opts.plot_dir = os.path.join(opts.result_root, opts.plot_dir)

try:
    os.makedirs(opts.result_root)
except OSError:
    if os.path.exists(opts.result_root):
        pass
    else:
        raise

if __name__ == '__main__':
    # Lists to keep losses
    student_losses_G = []
    teacher_losses_G = []
    list_centralityT = []
    list_centralityS = []
    list_pagerankT = []
    list_pagerankS = []
    list_eigenvectorT = []
    list_eigenvectorS = []
    list_student_loss_KL = []

    # For fast training
    cudnn.benchmark = True

    # ===============================COMPARISON METHODS AND L2S-KDnet================================================= #
    # 1. Baseline: we use a TS architecture where we do not include any adversarial regularization.
    # 2. Baseline+Discriminator: a variant of the first method where we add a discriminator for distinguishing
    # between the ground truth and predicted HR brain graphs.
    # 3. L2S-KDnet wo Local Topology: it is an ablated version of our L2S-KDnet framework which adopts only a
    # global topology loss function.
    # 4. L2S-KDnet wo TD regularization: in this method, we remove the teacher’s decoder from the training process of
    # the student network.
    # ================================================================================================================ #

    if opts.model == 'Baseline':
        TS_model = Baseline
    elif opts.model== 'Baseline w Discriminator':
        TS_model = BaselineWDiscriminatorForDecoder
    elif opts.model == 'L2S-KDnet wo TD regularization':
        TS_model = L2S_KDnet_withoutTDRegularization
    elif opts.model == 'L2S-KDnet wo Local Topology':
        TS_model = L2S_KDnet_withoutLocalTopology
    elif opts.model == 'L2S-KDnet':
        TS_model = L2S_KDnet
    else:
        raise Exception('Given mode parameter is invalid.')

    # creates output directories
    create_dirs_if_not_exist([opts.log_dir, opts.checkpoint_dir, opts.sample_dir, opts.result_dir, opts.plot_dir])

    with open(os.path.join(opts.result_root, 'opts.yaml'), 'w') as f:
        f.write(yaml.dump(vars(opts)))
        fold = 0

        '''
        ### simulated data generated with following codes:  ###
        real_morphological_data = np.random.normal(0.5, 0.1, (100, 595))
        real_functional_data = np.random.normal(0.5, 0.1, (100, 12720))
        '''

        # reads morphological LR data and functional HR data
        real_morphological_data = np.load(opts.lr)
        real_functional_data = np.load(opts.hr)

        # replaces negative values with zero
        real_functional_data[np.where(real_functional_data < 0)] = 0
        real_morphological_data[np.where(real_morphological_data < 0)] = 0

        kf = KFold(n_splits=opts.num_splits, shuffle=True, random_state=100)
        kf.get_n_splits(real_morphological_data)

        # performs train and test operations for each fold
        for train_index, test_index in kf.split(real_morphological_data):
            # takes fold train and test data
            X_train, X_test = real_morphological_data[train_index], real_morphological_data[test_index]
            y_train, y_test = real_functional_data[train_index], real_functional_data[test_index]

            real_morphological_loader = get_loader(X_train, X_train.shape[0], "train", opts.num_workers)
            real_functional_loader = get_loader(y_train, y_train.shape[0], "train", opts.num_workers)

            # initializes model
            model = TS_model(real_morphological_loader, real_functional_loader, opts)

            # training phase
            start_time = time.time()
            model.train(fold)
            training_duration = time.time() - start_time

            training_duration = str(datetime.timedelta(seconds=training_duration))[:-7]
            print("Training Duration:", training_duration)

            real_morphological_loader = get_loader(X_test, X_test.shape[0], "test", opts.num_workers)
            real_functional_loader = get_loader(y_test, y_test.shape[0], "test", opts.num_workers)

            # testing phase
            start_time = time.time()
            eigenvectorS, pagerankS, \
            eigenvectorT, pagerankT, \
            student_loss_G, teacher_loss_G, \
            student_loss_KL = model.test(real_morphological_loader, real_functional_loader, fold)

            testing_duration = time.time() - start_time
            testing_duration = str(datetime.timedelta(seconds=testing_duration))[:-7]
            print("Testing Duration:", testing_duration)

            # stores fold loss values
            student_losses_G.append(student_loss_G)
            teacher_losses_G.append(teacher_loss_G)
            list_pagerankT.append(pagerankT)
            list_pagerankS.append(pagerankS)
            list_eigenvectorT.append(eigenvectorT)
            list_eigenvectorS.append(eigenvectorS)
            list_student_loss_KL.append(student_loss_KL)

            # prints fold results
            printFoldResults(fold, teacher_loss_G, student_loss_G, pagerankT, pagerankS, eigenvectorT, eigenvectorS,student_loss_KL)

            fold += 1
        # prints final results
        printTestResults(student_losses_G, teacher_losses_G, list_pagerankT, list_pagerankS, list_eigenvectorT,list_eigenvectorS, list_student_loss_KL)
