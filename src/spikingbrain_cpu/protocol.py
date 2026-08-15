"""Small, defensive CPU/Gloo point-to-point protocol helpers."""
from __future__ import annotations
import fcntl, os, socket, struct
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist

HIDDEN_SHAPE = (1, 1, 3584)

def route_to(peer: str) -> dict[str, object]:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect((peer, 9)); local_ip = probe.getsockname()[0]
    finally: probe.close()
    interface = None
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for candidate in os.listdir("/sys/class/net"):
            try:
                packed = fcntl.ioctl(sock.fileno(), 0x8915, struct.pack("256s", candidate.encode()[:15]))
                if socket.inet_ntoa(packed[20:24]) == local_ip: interface = candidate; break
            except OSError: pass
    finally: sock.close()
    if interface is None: raise RuntimeError(f"cannot map route source {local_ip} to an interface")
    mtu = int(Path(f"/sys/class/net/{interface}/mtu").read_text())
    return {"interface": interface, "local_ip": local_ip, "peer": peer, "mtu": mtu}

def initialize(rank: int, master_addr: str, port: int, peer: str, timeout_seconds: int = 120):
    if not dist.is_available() or not dist.is_gloo_available(): raise RuntimeError("Gloo unavailable")
    route = route_to(peer)
    os.environ["GLOO_SOCKET_IFNAME"] = str(route["interface"])
    dist.init_process_group("gloo", init_method=f"tcp://{master_addr}:{port}", rank=rank, world_size=2, timeout=timedelta(seconds=timeout_seconds))
    return route

def validate_hidden(tensor: torch.Tensor) -> None:
    if tuple(tensor.shape) != HIDDEN_SHAPE: raise ValueError(f"unexpected hidden shape: {tuple(tensor.shape)}")
    if tensor.dtype != torch.float32: raise TypeError(f"unexpected hidden dtype: {tensor.dtype}")
    if not torch.isfinite(tensor).all(): raise FloatingPointError("hidden contains NaN/Inf")

def tensor_summary(tensor: torch.Tensor) -> dict[str, object]:
    flat=tensor.view(-1)
    return {"shape":list(tensor.shape),"dtype":str(tensor.dtype),"elements":tensor.numel(),"nbytes":tensor.numel()*tensor.element_size(),"norm":float(torch.linalg.vector_norm(tensor)),"min":float(tensor.min()),"max":float(tensor.max()),"sentinels":[float(flat[i]) for i in (0,1,17,1024,3583)]}

def shutdown() -> None:
    if dist.is_initialized():
        try: dist.barrier()
        finally: dist.destroy_process_group()
