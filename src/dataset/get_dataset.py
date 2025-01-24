from src.dataset.coreap_dataset import CoReaPDataset
from src.dataset.coreap_datacollator import CoReaPDataCollator

def get_dataset(args):

    if args.is_train:
        train_data = None
        valid_data = None 
        train_dataset = CoReaPDataset(args, train_data)
        valid_dataset = CoReaPDataset(args, valid_data)
        data_collator = CoReaPDataCollator(args)

        return train_dataset, valid_dataset, data_collator

    else:
        test_data = None
        test_dataset = CoReaPDataset(args, test_data)
        data_collator = CoReaPDataCollator(args)

        return test_dataset, None, data_collator


