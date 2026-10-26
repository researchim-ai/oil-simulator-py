"""Запись CUDA-графа без torch.cuda.graph: тот перед каждой записью вызывает gc.collect (~0.1 с)."""
import torch


def capture(fn, device):
    """Записывает fn() в CUDA-граф на устройстве device; возвращает (граф, результат fn в буферах графа)."""
    cur = torch.cuda.current_stream(device)
    side = torch.cuda.Stream(device=device)
    side.wait_stream(cur)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        graph.capture_begin()
        try:
            out = fn()
        finally:
            graph.capture_end()
    cur.wait_stream(side)
    return graph, out
