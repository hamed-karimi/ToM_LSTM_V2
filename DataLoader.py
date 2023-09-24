from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler, BatchSampler, Sampler
import numpy as np

from DataSet import AgentActionDataSet


def get_data_loader(utility):
    def get_generator_from_sampler(sampler: Sampler, batch_size):
        batch_sampler = BatchSampler(sampler=sampler,
                                     batch_size=batch_size,
                                     drop_last=False)
        params = {'batch_sampler': batch_sampler,
                  'pin_memory': False}

        generator = DataLoader(dataset, **params)
        return generator

    dataset = AgentActionDataSet(utility)
    train_batch_size = utility.params.BATCH_SIZE
    train_range = np.arange(int(utility.params.TRAIN_PROPORTION * len(dataset)))
    validation_range = np.arange(int(utility.params.TRAIN_PROPORTION * len(dataset)),
                                 int(utility.params.TRAIN_PROPORTION * len(dataset)) +
                                 int(utility.params.VALIDATION_PROPORTION * len(dataset)))
    test_range = np.arange(int(utility.params.TRAIN_PROPORTION * len(dataset)) +
                           int(utility.params.VALIDATION_PROPORTION * len(dataset)),
                           len(dataset))

    train_sampler = SubsetRandomSampler(train_range)
    validation_sampler = SequentialSampler(validation_range)
    test_sampler = SequentialSampler(test_range)

    train_generator = get_generator_from_sampler(train_sampler, batch_size=train_batch_size)
    validation_generator = get_generator_from_sampler(validation_sampler, batch_size=len(validation_range))
    test_generator = get_generator_from_sampler(test_sampler, batch_size=1)

    return train_generator, validation_generator, test_generator
