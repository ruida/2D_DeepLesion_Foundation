import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset, DeepLesionDataset


class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        elif self.dataset_name == 'deeplesion':
            self.dataset = DeepLesionDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        if self.dataset_name == 'deeplesion':
            collate = self.collate_fn_deeplesion
        else:
            collate = self.collate_fn

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        images_id, images, reports_ids, reports_masks, seq_lengths = zip(*data)
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks)


    @staticmethod
    def collate_fn_deeplesion(data):
        """
        Collate function for the 8-element DeepLesionDataset tuple.
        Returns:
          images_id, images, targets (LongTensor), targets_masks (FloatTensor),
          bbox_tensor (FloatTensor B x max_boxes x 4),
          bbox_mask   (BoolTensor  B x max_boxes),
          anatomy_tensor (LongTensor B x max_anatomy)
        """
        (images_id, images, reports_ids, reports_masks, seq_lengths,
         bbox_tensors, bbox_masks, anatomy_tensors) = zip(*data)

        images = torch.stack(images, 0)

        max_seq_length = max(seq_lengths)
        targets        = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks  = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        for i, (rids, rmsk) in enumerate(zip(reports_ids, reports_masks)):
            targets[i,       :len(rids)] = rids
            targets_masks[i, :len(rmsk)] = rmsk

        bbox_tensors    = torch.stack(bbox_tensors,    0)
        bbox_masks      = torch.stack(bbox_masks,      0)
        anatomy_tensors = torch.stack(anatomy_tensors, 0)

        return (
            images_id,
            images,
            torch.LongTensor(targets),
            torch.FloatTensor(targets_masks),
            bbox_tensors,
            bbox_masks,
            anatomy_tensors,
        )
