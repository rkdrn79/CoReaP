from src.datasets.dataset_512 import ImageFolderMaskDataset
from src.datasets.coreap_datacollator import CoReaPDataCollator

def get_dataset(args):

    if args.is_train:
        train_dataset = ImageFolderMaskDataset(path = './data/celeba_hq/train')
        valid_dataset = ImageFolderMaskDataset(path = './data/celeba_hq/val') 
        data_collator = CoReaPDataCollator(args)

        return train_dataset, valid_dataset, data_collator

    else:
        test_data = None
        test_dataset = CoReaPDataset(args, test_data)
        data_collator = CoReaPDataCollator(args)

        return test_dataset, None, data_collator


