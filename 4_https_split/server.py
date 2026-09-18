"""Stage 4 HTTPS server: middle decoder blocks B and server-side KV cache only."""

import json
import sys
import os

from https_core.settings import parser, select, project_path
from https_core.weights import prepare, RoleWeights
from https_core.engine import ServerState
from https_core.transport import LoopbackServer


def main(argv=None):
    argparser = parser(__doc__)
    argparser.add_argument("--instance-id", help="Optional launcher identity, not an authentication credential.")
    args = argparser.parse_args(argv)
    # 校验 instance_id
    if args.instance_id is not None and (
        len(args.instance_id) != 32 or any(c not in "0123456789abcdef" for c in args.instance_id)
    ):
        raise ValueError("instance-id must be 32 lowercase hex characters.")
    selected = select(args)
    if selected is None:
        return
    stage, runtime, profile, part = selected
    # 准备模型与身份信息
    directory, config, device, dtype, identity = prepare(profile, part, runtime)
    if stage["https"]["max_context_tokens"] > config.max_position_embeddings:
        raise ValueError("HTTPS context limit exceeds the model context.")
    print(f"Loading B only: layers {part.p + 1}..{part.p + part.k}, {device}, {dtype}.", flush=True)
    weights = RoleWeights(directory, config, part, "server", device, dtype)     # 加载模型权重
    net = stage["https"]    # 获取 HTTPS 网络配置
    state = ServerState(weights, identity, net, instance_id=args.instance_id)   # 创建服务端状态
    with LoopbackServer(state, project_path(net["credentials_dir"]), net["port"]) as server:
        print(json.dumps({
            "status": "ready", "pid": os.getpid(), "url": net["url"],
            "fingerprint": identity["fingerprint"], "parameters": weights.parameter_count,
            "weight_bytes": weights.weight_bytes, "layers_1based": [i + 1 for i in weights.indices],
            "full_model_loaded": False, "noise_std": 0.0,
        }), flush=True)
        print("Loopback only. No prompt/tensor logs. Ctrl+C stops the server and clears B caches.", flush=True)
        server.serve_forever(poll_interval=0.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Server stopped; sessions cleared.")
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
