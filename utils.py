import os
import time
import json
import datetime
import numpy as np
import tensorflow as tf
from collections import defaultdict, deque
from tensorflow.keras.callbacks import TensorBoard

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = np.array(self.deque)
        return np.median(d)

    @property
    def avg(self):
        d = np.array(self.deque, dtype=np.float32)
        return np.mean(d)

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return np.max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )

class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, tf.Tensor):
                v = v.numpy()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = [f"{name}: {str(meter)}" for name, meter in self.meters.items()]
        return self.delimiter.join(loss_str)

class TensorboardLogger:
    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        with self.writer.as_default():
            for k, v in kwargs.items():
                if v is None:
                    continue
                if isinstance(v, tf.Tensor):
                    v = v.numpy()
                assert isinstance(v, (float, int))
                tf.summary.scalar(f"{head}/{k}", v, step=self.step if step is None else step)
        self.writer.flush()

def seed_worker(worker_id):
    seed = int(tf.random.uniform([], maxval=2**32, dtype=tf.int32))
    np.random.seed(seed)
    random.seed(seed)

def save_model(args, epoch, model, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    checkpoint = {
        'model': model.get_weights(),
        'optimizer': optimizer.get_weights(),
        'epoch': epoch,
        'scaler': loss_scaler.get_weights(),
        'args': args.__dict__,
    }
    if model_ema is not None:
        checkpoint['model_ema'] = model_ema.get_weights()
    save_path = output_dir / f'checkpoint-{epoch_name}.ckpt'
    model.save_weights(save_path)
    with open(f'{save_path}.json', 'w') as f:
        json.dump(checkpoint, f)

def load_model(args, model, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint_path = output_dir / f'checkpoint-{args.resume}.ckpt'
        model.load_weights(checkpoint_path)
        with open(f'{checkpoint_path}.json', 'r') as f:
            checkpoint = json.load(f)
        optimizer.set_weights(checkpoint['optimizer'])
        loss_scaler.set_weights(checkpoint['scaler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if model_ema is not None:
            model_ema.set_weights(checkpoint['model_ema'])

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print(f"Set warmup steps = {warmup_iters}")
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array([final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * i / len(iters))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def multiple_samples_collate(batch, fold=False):
    inputs, labels, video_idx, extra_data = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    video_idx = [item for sublist in video_idx for item in sublist]
    inputs = tf.stack(inputs)
    labels = tf.stack(labels)
    video_idx = tf.stack(video_idx)
    if fold:
        return [inputs], labels, video_idx, extra_data
    else:
        return inputs, labels, video_idx, extra_data

def is_dist_avail_and_initialized():
    # TensorFlow doesn't have the same distributed utilities as PyTorch, so we assume single GPU for simplicity
    return False

def get_world_size():
    return 1

def get_rank():
    return 0

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        tf.saved_model.save(*args, **kwargs)

def init_distributed_mode(args):
    print('Not using distributed mode')
    args.distributed = False

# Example usage
class Args:
    def __init__(self, output_dir, resume):
        self.output_dir = output_dir
        self.resume = resume
        self.auto_resume = False
        self.start_epoch = 0

args = Args(output_dir='./checkpoints', resume='')

# Initialize model, optimizer, and loss scaler
model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
optimizer = tf.keras.optimizers.Adam()
loss_scaler = tf.keras.optimizers.schedules.ExponentialDecay(0.1, decay_steps=100000, decay_rate=0.96, staircase=True)

# Load model if resume
load_model(args, model, optimizer, loss_scaler)

# Save model checkpoint
save_model(args, 0, model, optimizer, loss_scaler)