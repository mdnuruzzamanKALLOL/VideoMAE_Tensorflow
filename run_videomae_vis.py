import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from timm.models import create_model
import utils
import modeling_pretrain
from datasets import DataAugmentationForVideoMAE
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from decord import VideoReader, cpu
from tensorflow.keras.applications import imagenet_utils

class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupCenterCrop(args.input_size)
        self.transform = tf.keras.Sequential([
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

def get_args():
    parser = argparse.ArgumentParser('VideoMAE visualization reconstruction script', add_help=False)
    parser.add_argument('img_path', type=str, help='input video path')
    parser.add_argument('save_path', type=str, help='save video path')
    parser.add_argument('model_path', type=str, help='checkpoint path of model')
    parser.add_argument('--mask_type', default='random', choices=['random', 'tube'],
                        type=str, help='masked strategy of video tokens/patches')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='depth of decoder')
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    parser.add_argument('--model', default='pretrain_videomae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    return parser.parse_args()

def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        decoder_depth=args.decoder_depth
    )
    return model

def main(args):
    print(args)

    device = tf.device(args.device)

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(args.model_path).expect_partial()
    model.eval()

    if args.save_path:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    with open(args.img_path, 'rb') as f:
        vr = VideoReader(f, ctx=cpu(0))
    duration = len(vr)
    new_length  = 1 
    new_step = 1
    skip_length = new_length * new_step
    
    tmp = np.arange(0,32, 2) + 60
    frame_id_list = tmp.tolist()

    video_data = vr.get_batch(frame_id_list).asnumpy()
    print(video_data.shape)
    img = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]

    transforms = DataAugmentationForVideoMAE(args)
    img, bool_masked_pos = transforms((img, None))
    img = tf.reshape(img, (args.num_frames, 3) + img.shape[-2:]).transpose(0, 1)
    bool_masked_pos = tf.convert_to_tensor(bool_masked_pos)

    with tf.GradientTape() as tape:
        img = tf.expand_dims(img, 0)
        print(img.shape)
        bool_masked_pos = tf.expand_dims(bool_masked_pos, 0)
        
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        bool_masked_pos = tf.convert_to_tensor(bool_masked_pos, dtype=tf.bool).flatten(1)
        outputs = model([img, bool_masked_pos], training=False)

        mean = tf.convert_to_tensor(IMAGENET_DEFAULT_MEAN)[None, :, None, None, None]
        std = tf.convert_to_tensor(IMAGENET_DEFAULT_STD)[None, :, None, None, None]
        ori_img = img * std + mean
        imgs = [Image.fromarray((ori_img[0,:,vid,:,:].numpy().clip(0, 1) * 255).astype(np.uint8)) for vid, _ in enumerate(frame_id_list)]
        for id, im in enumerate(imgs):
            im.save(f"{args.save_path}/ori_img{id}.jpg")

        img_squeeze = rearrange(ori_img, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size[0], p2=patch_size[0])
        img_norm = (img_squeeze - tf.reduce_mean(img_squeeze, axis=-2, keepdims=True)) / (tf.math.reduce_std(img_squeeze, axis=-2, keepdims=True) + 1e-6)
        img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
        img_patch[bool_masked_pos] = outputs

        mask = tf.ones_like(img_patch)
        mask[bool_masked_pos] = 0
        mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
        mask = rearrange(mask, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)', p0=2, p1=patch_size[0], p2=patch_size[1], h=14, w=14)

        rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
        rec_img = rec_img * (tf.math.reduce_std(img_squeeze, axis=-2, keepdims=True) + 1e-6) + tf.reduce_mean(img_squeeze, axis=-2, keepdims=True)
        rec_img = rearrange(rec_img, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)', p0=2, p1=patch_size[0], p2=patch_size[1], h=14, w=14)
        imgs = [Image.fromarray((rec_img[0, :, vid, :, :].numpy().clip(0, 1) * 255).astype(np.uint8)) for vid, _ in enumerate(frame_id_list)]
        for id, im in enumerate(imgs):
            im.save(f"{args.save_path}/rec_img{id}.jpg")

        img_mask = rec_img * mask
        imgs = [Image.fromarray((img_mask[0, :, vid, :, :].numpy().clip(0, 1) * 255).astype(np.uint8)) for vid, _ in enumerate(frame_id_list)]
        for id, im in enumerate(imgs):
            im.save(f"{args.save_path}/mask_img{id}.jpg")

if __name__ == '__main__':
    opts = get_args()
    main(opts)
