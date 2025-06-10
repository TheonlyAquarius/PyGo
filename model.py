import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Helper for Fixup Initialization
def fixup_initialize_conv(layer, num_blocks):
    nn.init.normal_(layer.weight, mean=0, std=math.sqrt(2 / (layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1] * num_blocks)))
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)

class GlobalSELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_fixup=True, num_blocks_fixup=1):
        super().__init__()
        self.use_fixup = use_fixup
        self.num_blocks_fixup = num_blocks_fixup
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=not use_fixup)
        self.relu = nn.ReLU(inplace=True)
        if self.use_fixup:
            self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
            fixup_initialize_conv(self.conv, self.num_blocks_fixup)
    def forward(self, x):
        x = self.conv(x)
        if self.use_fixup:
            x = x + self.bias
        return self.relu(x)

class StandardResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, use_fixup=True, num_blocks_fixup=1, use_se=False):
        super().__init__()
        self.use_fixup = use_fixup
        self.num_blocks_fixup = num_blocks_fixup
        self.bias1 = nn.Parameter(torch.zeros(1, channels, 1, 1)) if use_fixup else None
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=not use_fixup)
        self.bias2 = nn.Parameter(torch.zeros(1, channels, 1, 1)) if use_fixup else None
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=not use_fixup)
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1)) if use_fixup else None
        self.use_se = use_se
        if self.use_se:
            self.se = GlobalSELayer(channels)
        if self.use_fixup:
            fixup_initialize_conv(self.conv1, self.num_blocks_fixup)
            fixup_initialize_conv(self.conv2, self.num_blocks_fixup)
            nn.init.constant_(self.conv2.weight, 0)
    def forward(self, x):
        identity = x
        if self.use_fixup:
            out = x + self.bias1
        else:
            out = x
        out = F.relu(out)
        out = self.conv1(out)
        if self.use_fixup:
            out = out + self.bias2
        out = F.relu(out)
        out = self.conv2(out)
        if self.use_se:
            out = self.se(out)
        if self.use_fixup:
            out = out * self.scale
        return out + identity

class NestedBottleneckResidualBlock(nn.Module):
    def __init__(self, channels, bottleneck_factor=4, kernel_size=3, use_fixup=True, num_blocks_fixup=1, use_se=False):
        super().__init__()
        self.use_fixup = use_fixup
        self.num_blocks_fixup = num_blocks_fixup
        bottleneck_channels = channels // bottleneck_factor
        self.bias1 = nn.Parameter(torch.zeros(1, channels, 1, 1)) if use_fixup else None
        self.conv_reduce = nn.Conv2d(channels, bottleneck_channels, kernel_size=1, bias=not use_fixup)
        self.bias2 = nn.Parameter(torch.zeros(1, bottleneck_channels, 1, 1)) if use_fixup else None
        self.conv_main = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=not use_fixup)
        self.bias3 = nn.Parameter(torch.zeros(1, bottleneck_channels, 1, 1)) if use_fixup else None
        self.conv_expand = nn.Conv2d(bottleneck_channels, channels, kernel_size=1, bias=not use_fixup)
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1)) if use_fixup else None
        self.use_se = use_se
        if self.use_se:
            self.se = GlobalSELayer(channels)
        if self.use_fixup:
            fixup_initialize_conv(self.conv_reduce, self.num_blocks_fixup)
            fixup_initialize_conv(self.conv_main, self.num_blocks_fixup)
            fixup_initialize_conv(self.conv_expand, self.num_blocks_fixup)
            nn.init.constant_(self.conv_expand.weight, 0)
    def forward(self, x):
        identity = x
        if self.use_fixup:
            out = x + self.bias1
        else:
            out = x
        out = F.relu(out)
        out = self.conv_reduce(out)
        if self.use_fixup:
            out = out + self.bias2
        out = F.relu(out)
        out = self.conv_main(out)
        if self.use_fixup:
            out = out + self.bias3
        out = F.relu(out)
        out = self.conv_expand(out)
        if self.use_se:
            out = self.se(out)
        if self.use_fixup:
            out = out * self.scale
        return out + identity

class PolicyHead(nn.Module):
    def __init__(self, in_channels, policy_channels=8, mid_channels=32, board_size=19):
        super().__init__()
        self.board_size = board_size
        self.num_moves = board_size * board_size
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.policy_conv_move = nn.Conv2d(mid_channels, 1, kernel_size=1)
        self.pass_conv = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.pass_relu = nn.ReLU(inplace=True)
        self.pass_pool = nn.AdaptiveAvgPool2d(1)
        self.pass_fc = nn.Linear(4, 1)
    def forward(self, x, board_mask=None):
        pol_move_out = self.relu1(self.conv1(x))
        pol_move_logits = self.policy_conv_move(pol_move_out).squeeze(1)
        if board_mask is not None:
            assert board_mask.shape == pol_move_logits.shape, "board_mask must have the same shape as policy logits"
            assert board_mask.dtype == torch.float32, "board_mask must be of dtype float32"
            pol_move_logits = pol_move_logits.masked_fill(board_mask == 0, float('-inf'))
        pass_out = self.pass_relu(self.pass_conv(x))
        pass_out_pooled = self.pass_pool(pass_out).squeeze(-1).squeeze(-1)
        pass_logit = self.pass_fc(pass_out_pooled)
        move_logits_flat = pol_move_logits.view(pol_move_logits.size(0), -1)
        final_policy_logits = torch.cat([move_logits_flat, pass_logit], dim=1)
        return F.log_softmax(final_policy_logits, dim=1)

class ValueHead(nn.Module):
    def __init__(self, in_channels, mid_channels=32, fc_channels=256, board_size=19):
        super().__init__()
        self.board_size = board_size
        self.conv_val1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.relu_val1 = nn.ReLU(inplace=True)
        self.pool_val = nn.AdaptiveAvgPool2d(1)
        self.fc_val1 = nn.Linear(mid_channels, fc_channels)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc_game_outcome = nn.Linear(fc_channels, 3)
        self.fc_score_mean = nn.Linear(fc_channels, 1)
        self.fc_score_stdev = nn.Linear(fc_channels, 1)
        self.conv_own1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.relu_own1 = nn.ReLU(inplace=True)
        self.conv_own2 = nn.Conv2d(mid_channels, 1, kernel_size=1)
    def forward(self, x, board_mask=None):
        if board_mask is not None:
            assert board_mask.shape == (x.size(0), self.board_size, self.board_size), "board_mask must have the correct shape"
            assert board_mask.dtype == torch.float32, "board_mask must be of dtype float32"
        val_out = self.relu_val1(self.conv_val1(x))
        val_pooled = self.pool_val(val_out).squeeze(-1).squeeze(-1)
        fc_features = self.relu_fc1(self.fc_val1(val_pooled))
        game_outcome_logits = self.fc_game_outcome(fc_features)
        score_mean = self.fc_score_mean(fc_features)
        score_stdev = F.softplus(self.fc_score_stdev(fc_features)) + 1e-5
        own_out = self.relu_own1(self.conv_own1(x))
        ownership_map = torch.tanh(self.conv_own2(own_out))
        return game_outcome_logits, score_mean, score_stdev, ownership_map

class ModelConfig:
    def __init__(self, version_name="b18c384nbt-uec", in_channels=26, stem_channels=64, num_blocks=18, trunk_channels=384,
                 block_type="nbt", nbt_bottleneck_factor=4, policy_mid_channels=32, value_mid_channels=32, value_fc_channels=256,
                 board_size=19, use_fixup=True, use_se_in_trunk=False, use_one_batch_norm=True, use_global_bias=True):
        self.version_name = version_name
        self.in_channels = in_channels
        self.stem_channels = stem_channels
        self.num_blocks = num_blocks
        self.trunk_channels = trunk_channels
        self.block_type = block_type
        self.nbt_bottleneck_factor = nbt_bottleneck_factor
        self.policy_mid_channels = policy_mid_channels
        self.value_mid_channels = value_mid_channels
        self.value_fc_channels = value_fc_channels
        self.board_size = board_size
        self.use_fixup = use_fixup
        self.use_se_in_trunk = use_se_in_trunk
        self.use_one_batch_norm = use_one_batch_norm
        self.use_global_bias = use_global_bias

class KataGoModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_blocks_fixup = config.num_blocks * 2
        self.initial_block = InitialBlock(config.in_channels, config.stem_channels, use_fixup=config.use_fixup, num_blocks_fixup=self.num_blocks_fixup)
        if config.stem_channels != config.trunk_channels:
            self.stem_to_trunk = nn.Conv2d(config.stem_channels, config.trunk_channels, kernel_size=1, bias=not config.use_fixup)
            if config.use_fixup:
                fixup_initialize_conv(self.stem_to_trunk, self.num_blocks_fixup)
        else:
            self.stem_to_trunk = nn.Identity()
        trunk_blocks = []
        for _ in range(config.num_blocks):
            if config.block_type == "standard":
                block = StandardResidualBlock(config.trunk_channels, use_fixup=config.use_fixup, num_blocks_fixup=self.num_blocks_fixup, use_se=config.use_se_in_trunk)
            elif config.block_type == "nbt":
                block = NestedBottleneckResidualBlock(config.trunk_channels, bottleneck_factor=config.nbt_bottleneck_factor, use_fixup=config.use_fixup, num_blocks_fixup=self.num_blocks_fixup, use_se=config.use_se_in_trunk)
            else:
                raise ValueError(f"Unknown block type: {config.block_type}")
            trunk_blocks.append(block)
        self.trunk = nn.Sequential(*trunk_blocks)
        self.final_bn = nn.BatchNorm2d(config.trunk_channels) if config.use_one_batch_norm else None
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1) if config.use_global_bias else None
        self.fc_global = nn.Linear(config.trunk_channels, config.trunk_channels) if config.use_global_bias else None
        self.policy_head = PolicyHead(config.trunk_channels, config.policy_mid_channels, board_size=config.board_size)
        self.value_head = ValueHead(config.trunk_channels, config.value_mid_channels, config.value_fc_channels, board_size=config.board_size)
    def forward(self, x, board_mask=None, use_inference_heads=False):
        out = self.initial_block(x)
        out = self.stem_to_trunk(out)
        trunk_out = self.trunk(out)
        if self.config.use_one_batch_norm and not use_inference_heads:
            head_input = self.final_bn(trunk_out)
        else:
            head_input = trunk_out
        if self.config.use_global_bias:
            global_features = self.global_avg_pool(trunk_out).squeeze(-1).squeeze(-1)
            global_features = F.relu(self.fc_global(global_features))
            global_features = global_features.unsqueeze(-1).unsqueeze(-1).expand_as(head_input)
            head_input = head_input + global_features
        policy_logits = self.policy_head(head_input, board_mask)
        value_outputs = self.value_head(head_input, board_mask)
        game_outcome_logits, score_mean, score_stdev, ownership_map = value_outputs
        return {
            "policy_logits": policy_logits,
            "game_outcome_logits": game_outcome_logits,
            "score_mean": score_mean,
            "score_stdev": score_stdev,
            "ownership_map": ownership_map
        }

# Example Usage
if __name__ == '__main__':
    config_params = {
        "version_name": "b18c384nbt-uec-pytorch",
        "in_channels": 22,
        "stem_channels": 384,
        "num_blocks": 18,
        "trunk_channels": 384,
        "block_type": "nbt",
        "nbt_bottleneck_factor": 4,
        "policy_mid_channels": 32,
        "value_mid_channels": 32,
        "value_fc_channels": 256,
        "board_size": 19,
        "use_fixup": True,
        "use_se_in_trunk": True,
        "use_one_batch_norm": True,
        "use_global_bias": True
    }
    config = ModelConfig(**config_params)
    model = KataGoModel(config)
    batch_size = 2
    board_size = config.board_size
    dummy_input = torch.randn(batch_size, config.in_channels, board_size, board_size)
    dummy_mask = torch.ones(batch_size, board_size, board_size)
    dummy_mask[0, 0, :] = 0
    dummy_mask[0, :, 0] = 0
    print(f"Model: {config.version_name}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    model.train()
    outputs_train = model(dummy_input, board_mask=dummy_mask)
    print("\nTraining mode outputs:")
    print(f"  Policy logits shape: {outputs_train['policy_logits'].shape}")
    print(f"  Game outcome shape: {outputs_train['game_outcome_logits'].shape}")
    print(f"  Score mean shape: {outputs_train['score_mean'].shape}")
    print(f"  Score stdev shape: {outputs_train['score_stdev'].shape}")
    print(f"  Ownership map shape: {outputs_train['ownership_map'].shape}")
    print("\nPolicy logits for masked moves (should be -inf or very small):")
    print(f"  Logit at (0,0) for batch 0 (masked): {outputs_train['policy_logits'][0, 0]}")
    print(f"  Logit at (1,1) for batch 0 (unmasked): {outputs_train['policy_logits'][0, board_size + 1]}")
    model.eval()
    with torch.no_grad():
        outputs_infer = model(dummy_input, board_mask=dummy_mask, use_inference_heads=True)
    print("\nInference mode outputs:")
    print(f"  Policy logits shape: {outputs_infer['policy_logits'].shape}")
    print(f"  Ownership map shape: {outputs_infer['ownership_map'].shape}")
    print("\nPolicy logits for masked moves in inference mode (should be -inf or very small):")
    print(f"  Logit at (0,0) for batch 0 (masked): {outputs_infer['policy_logits'][0, 0]}")
