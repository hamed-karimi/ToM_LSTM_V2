from Train import train
from DataLoader import get_data_loader
from Test import test
import Utilities

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    utility = Utilities.Utilities()
    train_data_generator, validation_data_generator, test_data_generator = get_data_loader(utility)
    if utility.params.TRAINING_PHASE:
        train(train_data_generator, validation_data_generator)
    if utility.params.TESTING_PHASE:
        test(test_data_generator)
