import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import utils

def train_class_batch(model, samples, target, loss_fn):
    with tf.GradientTape() as tape:
        outputs = model(samples, training=True)
        loss = loss_fn(target, outputs)
    gradients = tape.gradient(loss, model.trainable_variables)
    return loss, outputs, gradients

def train_one_epoch(model, loss_fn, data_loader, optimizer, device, epoch, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 10

    for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = tf.convert_to_tensor(samples, dtype=tf.float32)
        targets = tf.convert_to_tensor(targets, dtype=tf.float32)

        loss, output, gradients = train_class_batch(model, samples, targets, loss_fn)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_value = loss.numpy()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        class_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=-1), tf.argmax(targets, axis=-1)), tf.float32)).numpy()

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)

        min_lr = min(param_group["lr"] for param_group in optimizer.param_groups)
        max_lr = max(param_group["lr"] for param_group in optimizer.param_groups)

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@tf.function
def validation_one_epoch(data_loader, model):
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    model.training = False

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        videos = tf.convert_to_tensor(videos, dtype=tf.float32)
        target = tf.convert_to_tensor(target, dtype=tf.int32)

        output = model(videos, training=False)
        loss = criterion(target, output)

        acc1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=-1), target), tf.float32)).numpy()
        acc5 = tf.reduce_mean(tf.cast(tf.reduce_any(tf.equal(tf.expand_dims(target, axis=-1), tf.nn.top_k(output, k=5).indices), axis=-1), tf.float32)).numpy()

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.numpy())
        metric_logger.meters['acc1'].update(acc1, n=batch_size)
        metric_logger.meters['acc5'].update(acc5, n=batch_size)

    metric_logger.synchronize_between_processes()
    print(f'* Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f}')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@tf.function
def final_test(data_loader, model, file):
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.training = False
    final_result = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        videos = tf.convert_to_tensor(videos, dtype=tf.float32)
        target = tf.convert_to_tensor(target, dtype=tf.int32)

        output = model(videos, training=False)
        loss = criterion(target, output)

        for i in range(output.shape[0]):
            string = f"{ids[i]} {output[i].numpy().tolist()} {int(target[i].numpy())} {int(chunk_nb[i].numpy())} {int(split_nb[i].numpy())}\n"
            final_result.append(string)

        acc1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=-1), target), tf.float32)).numpy()
        acc5 = tf.reduce_mean(tf.cast(tf.reduce_any(tf.equal(tf.expand_dims(target, axis=-1), tf.nn.top_k(output, k=5).indices), axis=-1), tf.float32)).numpy()

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.numpy())
        metric_logger.meters['acc1'].update(acc1, n=batch_size)
        metric_logger.meters['acc5'].update(acc5, n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write(f"{acc1}, {acc5}\n")
        for line in final_result:
            f.write(line)
    metric_logger.synchronize_between_processes()
    print(f'* Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f}')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
            data = tf.nn.softmax(data).numpy()
            if name not in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1, final_top5 = np.mean(top1), np.mean(top5)
    return final_top1 * 100, final_top5 * 100

def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
