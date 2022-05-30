import torch

from .dataclass import DataClass


class Model(DataClass):
    weight_sharing: bool = False
    checkpoint_path: str = "checkpoint.torch"
    steps_per_checkpoint: int = 0  # 0 -> disabled
    print_on_init: bool = True
    features: int = 256
    momentumnet_beta: float = 0.99  # The higher this is, the more numerically stable. BUT also lower impact per layer
    depth: int = 16
    batch_size: int = 32
    sequence_length: int = 32
    float16: bool = False
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    offloading: bool = False

    num_tokens:int = 256 # Dataset.classes,     
    dim:int = 8 # Embedding dimension = self.features.
    depth:int = 1
    max_seq_len: int = 32
    heads:int = 8
    dim_head:int = 8
    causal:bool = True
    emb_dim:int = 16 # also embedding dimension
    reversible:bool = False
    ff_chunks:int = 1
    ff_glu:bool = False
    ff_dropout = 0.
    attn_layer_dropout = 0.
    attn_dropout = 0.
    blindspot_size = 1
    n_local_attn_heads = 0
    local_attn_window_size = 32
    return_embeddings = False
    receives_context = False
    pkm_layers = tuple()
    pkm_num_keys = 32
    attend_axially = False
    linformer_settings = None
    context_linformer_settings = None
    use_axial_pos_emb = False
    use_rotary_emb = False
    shift_tokens = False
    input_embedding_std: float = 1.
    position_embedding_std: float = 1.