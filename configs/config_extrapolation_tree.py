from dataclasses import dataclass, field

@dataclass
class GeneralConfig:
    seed: int = 42
    use_wandb: bool = True
    save_path: str = "results/"
    device: str = "cuda"

@dataclass
class SamplerConfig:
    use_ema: bool = False
    probability_flow: bool = False
    noise_removal: bool = True
    eps: float = 1e-4
    solver: str = "S4"
    predictor: str = "Euler"    # Options: "Euler", "S4"
    corrector: str = "Langevin"    # Options: "None", "Langevin", etc.
    snr: float = 0.05
    scale_eps: float = 0.7
    n_steps: int = 1
    num_nodes: int = 30

@dataclass
class DataConfig:
    dir: str = "data"
    data: str = "extrapolation_tree"
    batch_size: int = 128
    max_node_num: int = 40
    max_feat_num: int = 1
    init: str = "ones"
    test_split: float = 0.2
    val_split: float = 0.1

@dataclass
class ModelConfig:
    max_feat_num: int = 1
    nhid: int = 32
    num_layers: int = 7
    num_linears: int = 2
    c_init: int = 2
    c_hid: int = 8
    c_final: int = 4
    adim: int = 32
    num_heads: int = 4
    conv: str = "GCN"

@dataclass
class TrainConfig:
    lr: float = 0.01
    weight_decay: float = 0.0001
    ema: float = 0.999
    eps: float = 1e-5
    reduce_mean: bool = False
    lr_schedule: bool = True
    lr_decay: float = 0.999
    num_epochs: int = 5000
    grad_norm: float = 1.0

@dataclass
class SDEConfig:
    type: str = "VP"
    beta_min: float = 0.1
    beta_max: float = 1.0
    num_scales: int = 1000

@dataclass
class MainConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sde: SDEConfig = field(default_factory=SDEConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)