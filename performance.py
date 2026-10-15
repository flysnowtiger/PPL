"""Synchronized runtime measurements for training and evaluation."""

import time
from dataclasses import dataclass
from typing import Optional

import torch


def synchronize(use_gpu):
    """Wait for queued CUDA work before reading a wall-clock timer."""
    if use_gpu:
        torch.cuda.synchronize()


def reset_peak_gpu_memory(use_gpu):
    if use_gpu:
        torch.cuda.reset_peak_memory_stats()


def peak_gpu_memory_mb(use_gpu):
    if not use_gpu:
        return None
    return torch.cuda.max_memory_allocated() / (1024**2)


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def format_elapsed(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def print_model_flops(model, sequence_length, image_size):
    """Print FLOPs for one configured video sequence without changing weights."""
    try:
        from thop import profile
    except ImportError:
        print("Model FLOPs: unavailable (install the 'thop' package)")
        return

    profiled_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    device = next(profiled_model.parameters()).device
    height, width = image_size
    inputs = torch.zeros(1, sequence_length, 3, height, width, device=device)
    training_states = {module: module.training for module in profiled_model.modules()}
    thop_attributes = {
        module: (hasattr(module, "total_ops"), hasattr(module, "total_params"))
        for module in profiled_model.modules()
    }
    profiled_model.eval()
    try:
        with torch.no_grad():
            macs, _ = profile(profiled_model, inputs=(inputs,), verbose=False)
        print(
            "Model FLOPs: {:.5f} G per video sequence "
            "(SEQ_LEN={}, input={}x{}, 1 MAC = 1 FLOP)".format(
                macs / 1e9,
                sequence_length,
                height,
                width,
            )
        )
    except Exception as error:
        print("Model FLOPs: unavailable ({})".format(error))
    finally:
        for module, state in training_states.items():
            module.training = state
        for module, (had_total_ops, had_total_params) in thop_attributes.items():
            if not had_total_ops and hasattr(module, "total_ops"):
                delattr(module, "total_ops")
            if not had_total_params and hasattr(module, "total_params"):
                delattr(module, "total_params")


@dataclass
class InferenceMetrics:
    total_forward_time_seconds: float
    total_extraction_time_seconds: float
    measured_batches: int
    measured_videos: int
    measured_sequences: int
    measured_frames: int
    latency_ms_per_video: float
    latency_p50_ms_per_video: float
    latency_p95_ms_per_video: float
    throughput_videos_per_second: float
    latency_ms_per_sequence: float
    latency_p50_ms_per_sequence: float
    latency_p95_ms_per_sequence: float
    throughput_sequences_per_second: float
    throughput_frames_per_second: float
    peak_gpu_memory_mb: Optional[float]


class InferenceProfiler:
    """Measure synchronized model-forward time without changing inference."""

    def __init__(self, use_gpu, warmup_batches=5):
        self.use_gpu = use_gpu
        self.warmup_batches = max(0, warmup_batches)
        self.records = []
        self.extraction_seconds = 0.0
        self._span_start = None

    def start(self):
        reset_peak_gpu_memory(self.use_gpu)

    def start_extraction(self):
        synchronize(self.use_gpu)
        self._span_start = time.perf_counter()

    def stop_extraction(self):
        synchronize(self.use_gpu)
        self.extraction_seconds += time.perf_counter() - self._span_start
        self._span_start = None

    def measure(self, forward, video_count, sequence_count, frame_count):
        synchronize(self.use_gpu)
        start = time.perf_counter()
        outputs = forward()
        synchronize(self.use_gpu)
        elapsed = time.perf_counter() - start
        self.records.append((elapsed, video_count, sequence_count, frame_count))
        return outputs

    def summarize(self):
        measured = self.records[self.warmup_batches :]
        if not measured:
            measured = self.records

        forward_seconds = sum(record[0] for record in measured)
        videos = sum(record[1] for record in measured)
        sequences = sum(record[2] for record in measured)
        frames = sum(record[3] for record in measured)
        per_video_ms = [
            elapsed * 1000.0 / max(video_count, 1)
            for elapsed, video_count, _, _ in measured
        ]
        per_sequence_ms = [
            elapsed * 1000.0 / max(sequence_count, 1)
            for elapsed, _, sequence_count, _ in measured
        ]
        denominator = max(forward_seconds, 1e-12)
        return InferenceMetrics(
            total_forward_time_seconds=forward_seconds,
            total_extraction_time_seconds=self.extraction_seconds,
            measured_batches=len(measured),
            measured_videos=videos,
            measured_sequences=sequences,
            measured_frames=frames,
            latency_ms_per_video=forward_seconds * 1000.0 / max(videos, 1),
            latency_p50_ms_per_video=percentile(per_video_ms, 0.50),
            latency_p95_ms_per_video=percentile(per_video_ms, 0.95),
            throughput_videos_per_second=videos / denominator,
            latency_ms_per_sequence=forward_seconds * 1000.0 / max(sequences, 1),
            latency_p50_ms_per_sequence=percentile(per_sequence_ms, 0.50),
            latency_p95_ms_per_sequence=percentile(per_sequence_ms, 0.95),
            throughput_sequences_per_second=sequences / denominator,
            throughput_frames_per_second=frames / denominator,
            peak_gpu_memory_mb=peak_gpu_memory_mb(self.use_gpu),
        )


def print_inference_metrics(metrics):
    peak_memory = metrics.peak_gpu_memory_mb or 0.0
    print(
        "Inference time (query + gallery feature extraction): "
        f"{metrics.total_extraction_time_seconds:.3f}s "
        f"({format_elapsed(metrics.total_extraction_time_seconds)})"
    )
    print(
        "Inference performance: "
        f"profiled forward time {metrics.total_forward_time_seconds:.3f}s, "
        f"video latency {metrics.latency_ms_per_video:.3f} ms/video, "
        f"video P50 {metrics.latency_p50_ms_per_video:.3f} ms, "
        f"video P95 {metrics.latency_p95_ms_per_video:.3f} ms, "
        f"video throughput {metrics.throughput_videos_per_second:.3f} videos/s, "
        f"clip latency {metrics.latency_ms_per_sequence:.3f} ms/clip, "
        f"clip P50 {metrics.latency_p50_ms_per_sequence:.3f} ms, "
        f"clip P95 {metrics.latency_p95_ms_per_sequence:.3f} ms, "
        f"clip throughput {metrics.throughput_sequences_per_second:.3f} clips/s, "
        f"{metrics.throughput_frames_per_second:.3f} frames/s, "
        f"peak GPU memory {peak_memory:.2f} MiB"
    )
