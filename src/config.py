
# CIFAR ResNet config
CIFAR_CONFIG = {
    "num_classes": 10,
    "in_channels": 16,
    "block_counts": [2, 2, 2, 2],       # stage başına residual block sayısı
    "channels_per_stage": [16, 32, 64, 64],  # her stage'deki çıkış kanalları
    "survival_prob_start": 1.0,          # ilk blok için
    "survival_prob_end": 0.5,            # son blok için
}


# ImageNet ResNet config

IMAGENET_CONFIG = {
    "num_classes": 1000,
    "in_channels": 64,
    "block_counts": [3, 4, 6, 3],        # ResNet-50 örneği
    "channels_per_stage": [256, 512, 1024, 2048],
    "survival_prob_start": 1.0,
    "survival_prob_end": 0.5,
    "initial_conv_kernel": 7,
    "initial_conv_stride": 2,
    "initial_conv_padding": 3,
}
