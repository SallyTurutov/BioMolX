import torch
import random


def obtain_distr_list(dataset):
    if dataset == 'brain':
        #       human       mouse     rat          pig       guinea pig  bovine      rabbit
        return [[153, 163], [19, 22], [2770, 706], [41, 70], [235, 431], [316, 114], [103, 1]]
    elif dataset == 'breast':
        #       human
        return [[18351, 3499]]
    elif dataset == 'cervix':
        #       human
        return [[24244, 1472]]
    elif dataset == 'intestinal':
        #       human       guinea pig
        return [[144, 291], [28, 10]]
    elif dataset == 'kidney':
        #       human          rat         pig
        return [[56919, 2491], [153, 274], [19, 1]]
    elif dataset == 'liver':
        #       human           rat
        return [[134910, 8461], [3626, 274]]
    elif dataset == 'lung':
        #       human          guinea pig
        return [[12888, 1632], [79, 30]]
    elif dataset == 'ovary':
        #       human          chinese hamster
        return [[14096, 1856], [1640, 148]]
    elif dataset == 'prostate':
        #       human        rat
        return [[2689, 843], [238, 156]]
    elif dataset == 'skin':
        #       human
        return [[18847, 2933]]


def sample_datasets(data, dataset, task, n_query, epoch=0):
    print('Sampling task: ' + str(task) + ' with len: ' + str(len(data)))

    if dataset == 'lung' or dataset == 'ovary' or dataset == 'liver':
        # There must be positive and negative samples in order to calc AUC, so we use a different seed
        random.seed(32)
    else:
        random.seed(42)

    query_list = random.sample(range(0, len(data)), n_query)
    support_list = [i for i in range(0, len(data)) if i not in query_list]

    ##############################################
    # query_list = query_list[:-1] + [len(data) - 1]
    # use this for brain and rabbit as target - otherwise, it won't find the 1 positive sample
    ##############################################

    random.seed(epoch)
    random.shuffle(query_list)
    random.shuffle(support_list)

    support_dataset = data[torch.tensor(support_list)]
    query_dataset = data[torch.tensor(query_list)]

    return support_dataset, query_dataset



