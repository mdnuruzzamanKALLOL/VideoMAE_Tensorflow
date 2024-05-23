import math
import tensorflow as tf
from tensorflow.keras import layers, models
from functools import partial

class DropPath(layers.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if not training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        input_shape = tf.shape(x)
        random_tensor = keep_prob + tf.random.uniform(input_shape, dtype=x.dtype)
        random_tensor = tf.floor(random_tensor)
        output = x / keep_prob * random_tensor
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})
        return config

class Mlp(layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = layers.Dense(hidden_features)
        self.act = layers.Activation('gelu')
        self.fc2 = layers.Dense(out_features)
        self.drop = layers.Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(layers.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = layers.Dense(all_head_dim * 3, use_bias=qkv_bias)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)

    def call(self, x):
        B, N, C = tf.shape(x)
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (B, N, 3, self.num_heads, -1))
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2] 

        q = q * self.scale
        attn = tf.matmul(q, k, transpose_b=True)

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (B, N, -1))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(layers.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer='gelu', norm_layer='layer_norm', attn_head_dim=None):
        super(Block, self).__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else layers.Activation('linear')
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.gamma_1 = tf.Variable(init_values * tf.ones((dim,)), trainable=True) if init_values > 0 else None
        self.gamma_2 = tf.Variable(init_values * tf.ones((dim,)), trainable=True) if init_values > 0 else None

    def call(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class PatchEmbed(layers.Layer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.proj = layers.Conv3D(embed_dim, kernel_size=(tubelet_size, patch_size, patch_size), strides=(tubelet_size, patch_size, patch_size))

    def call(self, x):
        B, T, H, W, C = tf.shape(x)
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        x = self.proj(x)
        x = tf.reshape(x, (B, -1, tf.shape(x)[-1]))
        return x

def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return tf.convert_to_tensor(sinusoid_table, dtype=tf.float32)

class PretrainVisionTransformerEncoder(models.Model):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer='layer_norm', init_values=None, tubelet_size=2, use_checkpoint=False,
                 use_learnable_pos_emb=False):
        super(PretrainVisionTransformerEncoder, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint

        if use_learnable_pos_emb:
            self.pos_embed = self.add_weight('pos_embed', shape=[1, num_patches, embed_dim], initializer='zeros')
        else:
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = np.linspace(0, drop_path_rate, depth)
        self.blocks = [Block(embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[i], init_values, norm_layer) for i in range(depth)]
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.head = layers.Dense(num_classes) if num_classes > 0 else layers.Activation('linear')

    def call(self, x, mask):
        _, _, T, _, _ = tf.shape(x)
        x = self.patch_embed(x)
        x += tf.cast(self.pos_embed, x.dtype)

        B, _, C = tf.shape(x)
        x_vis = tf.reshape(x[~mask], (B, -1, C))

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

class PretrainVisionTransformerDecoder(models.Model):
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer='layer_norm', init_values=None, num_patches=196, tubelet_size=2, use_checkpoint=False):
        super(PretrainVisionTransformerDecoder, self).__init__()
        self.num_classes = num_classes
        assert num_classes == 3 * tubelet_size * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint

        dpr = np.linspace(0, drop_path_rate, depth)
        self.blocks = [Block(embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[i], init_values, norm_layer) for i in range(depth)]
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.head = layers.Dense(num_classes) if num_classes > 0 else layers.Activation('linear')

    def call(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:]))
        else:
            x = self.head(self.norm(x))

        return x

class PretrainVisionTransformer(models.Model):
    def __init__(self, img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0, encoder_embed_dim=768, encoder_depth=12,
                 encoder_num_heads=12, decoder_num_classes=1536, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer='layer_norm',
                 init_values=0., use_learnable_pos_emb=False, use_checkpoint=False, tubelet_size=2, num_classes=0, in_chans=0):
        super(PretrainVisionTransformer, self).__init__()
        self.encoder = PretrainVisionTransformerEncoder(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, num_classes=encoder_num_classes,
                                                        embed_dim=encoder_embed_dim, depth=encoder_depth, num_heads=encoder_num_heads, mlp_ratio=mlp_ratio,
                                                        qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                                        norm_layer=norm_layer, init_values=init_values, tubelet_size=tubelet_size, use_checkpoint=use_checkpoint,
                                                        use_learnable_pos_emb=use_learnable_pos_emb)

        self.decoder = PretrainVisionTransformerDecoder(patch_size=patch_size, num_patches=self.encoder.patch_embed.num_patches,
                                                        num_classes=decoder_num_classes, embed_dim=decoder_embed_dim, depth=decoder_depth, num_heads=decoder_num_heads,
                                                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                        drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values, tubelet_size=tubelet_size,
                                                        use_checkpoint=use_checkpoint)

        self.encoder_to_decoder = layers.Dense(decoder_embed_dim, use_bias=False)

        self.mask_token = self.add_weight('mask_token', shape=[1, 1, decoder_embed_dim], initializer='zeros')
        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)

    def call(self, x, mask):
        _, _, T, _, _ = tf.shape(x)
        x_vis = self.encoder(x, mask)
        x_vis = self.encoder_to_decoder(x_vis)
        B, N, C = tf.shape(x_vis)

        expand_pos_embed = tf.cast(self.pos_embed, x.dtype)
        pos_emd_vis = tf.reshape(expand_pos_embed[~mask], (B, -1, C))
        pos_emd_mask = tf.reshape(expand_pos_embed[mask], (B, -1, C))
        x_full = tf.concat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], axis=1)
        x = self.decoder(x_full, tf.shape(pos_emd_mask)[1])

        return x

def pretrain_videomae_small_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(img_size=224, patch_size=16, encoder_embed_dim=384, encoder_depth=12, encoder_num_heads=6,
                                      encoder_num_classes=0, decoder_num_classes=1536, decoder_embed_dim=192, decoder_num_heads=3,
                                      mlp_ratio=4, qkv_bias=True, norm_layer=partial(layers.LayerNormalization, epsilon=1e-6), **kwargs)
    if pretrained:
        checkpoint = tf.train.load_checkpoint(kwargs["init_ckpt"])
        model.load_weights(checkpoint)
    return model

def pretrain_videomae_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(img_size=224, patch_size=16, encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                                      encoder_num_classes=0, decoder_num_classes=1536, decoder_embed_dim=384, decoder_num_heads=6,
                                      mlp_ratio=4, qkv_bias=True, norm_layer=partial(layers.LayerNormalization, epsilon=1e-6), **kwargs)
    if pretrained:
        checkpoint = tf.train.load_checkpoint(kwargs["init_ckpt"])
        model.load_weights(checkpoint)
    return model

def pretrain_videomae_large_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(img_size=224, patch_size=16, encoder_embed_dim=1024, encoder_depth=24, encoder_num_heads=16,
                                      encoder_num_classes=0, decoder_num_classes=1536, decoder_embed_dim=512, decoder_num_heads=8,
                                      mlp_ratio=4, qkv_bias=True, norm_layer=partial(layers.LayerNormalization, epsilon=1e-6), **kwargs)
    if pretrained:
        checkpoint = tf.train.load_checkpoint(kwargs["init_ckpt"])
        model.load_weights(checkpoint)
    return model

def pretrain_videomae_huge_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(img_size=224, patch_size=16, encoder_embed_dim=1280, encoder_depth=32, encoder_num_heads=16,
                                      encoder_num_classes=0, decoder_num_classes=1536, decoder_embed_dim=640, decoder_num_heads=8,
                                      mlp_ratio=4, qkv_bias=True, norm_layer=partial(layers.LayerNormalization, epsilon=1e-6), **kwargs)
    if pretrained:
        checkpoint = tf.train.load_checkpoint(kwargs["init_ckpt"])
        model.load_weights(checkpoint)
    return model
