"""
trainer_deeplesion.py
=====================

Trainer for the DeepLesion version of R2Gen-Mamba.

Expected DeepLesion batch format:
    (
        images_id,
        images,
        targets,
        targets_masks,
        bboxes,
        bbox_mask,
        anatomy_ids,
    )

Important:
    anatomy_ids here can mean different things depending on --anatomy_source:

    --anatomy_source old
        Uses original anatomy_ids from deeplesion_mamba_final.json.
        This is leakage-prone and can produce artificially high BLEU.

    --anatomy_source rough
        Uses one coarse rough_anatomy_id from
        deeplesion_mamba_final_rough_anatomy.json.

    --anatomy_source none
        Uses all-zero anatomy tensor / effectively no anatomy guidance.

This trainer does not decide which anatomy source to use.
That is handled in modules/datasets.py.
"""

import torch
from modules.trainer import BaseTrainer


class DeepLesionTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        args,
        lr_scheduler,
        train_dataloader,
        val_dataloader,
        test_dataloader,
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, args)

        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _to_device(self, batch):
        """
        Move one DeepLesion batch to GPU/CPU device.

        Batch format:
            images_id:      list[str]
            images:         Tensor
            targets:        Tensor
            targets_masks:  Tensor
            bboxes:         Tensor, shape [B, max_boxes, 4]
            bbox_mask:      Tensor, shape [B, max_boxes]
            anatomy_ids:    Tensor, shape [B, max_anatomy]
        """
        (
            images_id,
            images,
            targets,
            targets_masks,
            bboxes,
            bbox_mask,
            anatomy_ids,
        ) = batch

        images = images.to(self.device)
        targets = targets.to(self.device)
        targets_masks = targets_masks.to(self.device)
        bboxes = bboxes.to(self.device)
        bbox_mask = bbox_mask.to(self.device)
        anatomy_ids = anatomy_ids.to(self.device)

        return (
            images_id,
            images,
            targets,
            targets_masks,
            bboxes,
            bbox_mask,
            anatomy_ids,
        )

    def _run_eval_loop(self, dataloader):
        """
        Run true autoregressive/sampling evaluation.

        Important:
            Prediction is decoded from model output.
            Ground truth is decoded from target tokens.
            This avoids the bug where GT is accidentally used as prediction.
        """
        self.model.eval()

        gts = []
        res = []

        with torch.no_grad():
            for batch in dataloader:
                (
                    images_id,
                    images,
                    targets,
                    targets_masks,
                    bboxes,
                    bbox_mask,
                    anatomy_ids,
                ) = self._to_device(batch)

                output = self.model(
                    images,
                    mode="sample",
                    bboxes=bboxes,
                    bbox_mask=bbox_mask,
                    anatomy_ids=anatomy_ids,
                )

                pred_reports = self.model.tokenizer.decode_batch(
                    output.cpu().numpy()
                )

                gt_reports = self.model.tokenizer.decode_batch(
                    targets[:, 1:].cpu().numpy()
                )

                res.extend(pred_reports)
                gts.extend(gt_reports)

        metrics = self.metric_ftns(
            {i: [gt] for i, gt in enumerate(gts)},
            {i: [pred] for i, pred in enumerate(res)},
        )

        return metrics

    def _train_epoch(self, epoch):
        """
        Train one epoch, then evaluate on validation and test sets.
        """
        self.model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(self.train_dataloader):
            (
                images_id,
                images,
                targets,
                targets_masks,
                bboxes,
                bbox_mask,
                anatomy_ids,
            ) = self._to_device(batch)

            output = self.model(
                images,
                targets=targets,
                mode="train",
                bboxes=bboxes,
                bbox_mask=bbox_mask,
                anatomy_ids=anatomy_ids,
            )

            loss = self.criterion(output, targets, targets_masks)

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)

            self.optimizer.step()

            train_loss += loss.item()

        log = {
            "train_loss": train_loss / len(self.train_dataloader)
        }

        val_metrics = self._run_eval_loop(self.val_dataloader)
        log.update(
            {
                "val_" + key: value
                for key, value in val_metrics.items()
            }
        )

        test_metrics = self._run_eval_loop(self.test_dataloader)
        log.update(
            {
                "test_" + key: value
                for key, value in test_metrics.items()
            }
        )

        self.lr_scheduler.step()

        return log
