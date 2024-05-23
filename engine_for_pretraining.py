import math
import sys
from typing import Iterable
import tensorflow as tf
import utils
from einops import rearrange

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

def train_one_epoch(model, data_loader: Iterable, optimizer: tf.keras.optimizers.Optimizer,
                    epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normalize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 10

    loss_func = tf.keras.losses.MeanSquaredError()

    for step, (videos, bool_masked_pos) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for param_group in optimizer.param_groups:
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos = tf.convert_to_tensor(videos, dtype=tf.float32)
        bool_masked_pos = tf.convert_to_tensor(bool_masked_pos, dtype=tf.bool)

        mean = tf.constant(IMAGENET_DEFAULT_MEAN, shape=[1, 1, 1, 1, 3])
        std = tf.constant(IMAGENET_DEFAULT_STD, shape=[1, 1, 1, 1, 3])
        unnorm_videos = videos * std + mean  # in [0, 1]

        if normalize_target:
            videos_squeeze = rearrange(unnorm_videos, 'b t (h p1) (w p2) c -> b (t h w) (p1 p2) c', p1=patch_size, p2=patch_size)
            videos_norm = (videos_squeeze - tf.reduce_mean(videos_squeeze, axis=-2, keepdims=True)) / (tf.math.reduce_std(videos_squeeze, axis=-2, keepdims=True) + 1e-6)
            videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
        else:
            videos_patch = rearrange(unnorm_videos, 'b t (h p1) (w p2) c -> b (t h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

        B, _, C = videos_patch.shape
        labels = tf.boolean_mask(videos_patch, bool_masked_pos).reshape(B, -1, C)

        with tf.GradientTape() as tape:
            outputs = model([videos, bool_masked_pos], training=True)
            loss = loss_func(labels, outputs)

        loss_value = loss.numpy()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        metric_logger.update(loss=loss_value)

        lr = optimizer.learning_rate
        min_lr = min(lr) if isinstance(lr, list) else lr
        max_lr = max(lr) if isinstance(lr, list) else lr
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        weight_decay_value = optimizer.weight_decay if hasattr(optimizer, 'weight_decay') else None
        metric_logger.update(weight_decay=weight_decay_value)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
