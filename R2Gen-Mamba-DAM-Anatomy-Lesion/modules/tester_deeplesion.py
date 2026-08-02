"""
tester_deeplesion.py
====================
Tester subclass for DeepLesion.  Mirrors the original Tester but
unpacks the 7-element batch and passes bbox / anatomy to the model.

Usage
-----
    from modules.tester_deeplesion import DeepLesionTester

    tester = DeepLesionTester(
        model=model, criterion=compute_loss,
        metric_ftns=compute_scores, args=args,
        test_dataloader=test_dl,
    )
    tester.test()
"""

import os
import logging

import pandas as pd
import torch
from tqdm import tqdm

from modules.tester import BaseTester


class DeepLesionTester(BaseTester):
    def __init__(self, model, criterion, metric_ftns, args, test_dataloader):
        super().__init__(model, criterion, metric_ftns, args)
        self.test_dataloader = test_dataloader

    # ------------------------------------------------------------------
    def _to_device(self, batch):
        (images_id, images, targets, targets_masks,
         bboxes, bbox_mask, anatomy_ids) = batch

        images        = images.to(self.device)
        targets       = targets.to(self.device)
        targets_masks = targets_masks.to(self.device)
        bboxes        = bboxes.to(self.device)
        bbox_mask     = bbox_mask.to(self.device)
        anatomy_ids   = anatomy_ids.to(self.device)

        return (images_id, images, targets, targets_masks,
                bboxes, bbox_mask, anatomy_ids)

    # ------------------------------------------------------------------
    def test(self):
        self.logger.info('Start evaluating on the DeepLesion test set.')
        log = {}
        self.model.eval()

        with torch.no_grad():
            test_gts, test_res = [], []

            for batch in tqdm(self.test_dataloader,
                              desc='Testing', unit='batch'):
                (images_id, images, targets, targets_masks,
                 bboxes, bbox_mask, anatomy_ids) = self._to_device(batch)

                output = self.model(
                    images,
                    mode        = 'sample',
                    bboxes      = bboxes,
                    bbox_mask   = bbox_mask,
                    anatomy_ids = anatomy_ids,
                )
                reports       = self.model.tokenizer.decode_batch(
                    output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(
                    targets[:, 1:].cpu().numpy())

                test_res.extend(reports)
                test_gts.extend(ground_truths)

        test_met = self.metric_ftns(
            {i: [gt] for i, gt in enumerate(test_gts)},
            {i: [re] for i, re in enumerate(test_res)},
        )
        log.update(**{'test_' + k: v for k, v in test_met.items()})
        print(log)

        # Save predictions and references to CSV
        pd.DataFrame(test_res).to_csv(
            os.path.join(self.save_dir, 'res.csv'), index=False, header=False)
        pd.DataFrame(test_gts).to_csv(
            os.path.join(self.save_dir, 'gts.csv'), index=False, header=False)

        return log

    # ------------------------------------------------------------------
    def plot(self):
        """Attention-heatmap visualisation – not implemented for DeepLesion."""
        self.logger.warning(
            'plot() is not implemented for DeepLesionTester. '
            'Use the original Tester.plot() for attention visualisation.')
