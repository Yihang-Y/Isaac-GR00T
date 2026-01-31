from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gr00t.data.interfaces import ShardedDataset
from gr00t.data.types import EmbodimentTag, MessageType, ModalityConfig, VLAStepData

from .lerobot_episode_loader import LeRobotEpisodeLoader
from .sharded_single_step_dataset import extract_step_data


class ShardedChunkDataset(ShardedDataset):
    """
    Chunk-level dataset that creates shards from continuous sequence windows within episodes.

    This dataset implementation provides chunk-level data access for memory-aware VLA training by:
    1. Loading episodes using LeRobotEpisodeLoader
    2. Generating continuous windows of length chunk_len from each episode
    3. Organizing chunks into balanced shards for efficient loading
    4. Supporting episode subsampling for data efficiency

    Each chunk contains:
    - episode_id: Identifier for the source episode
    - start_t: Starting timestep index within the episode
    - steps: List of L consecutive timesteps (obs[t:t+L], act[t:t+L], mask)

    Key features:
    - Chunk-level data access (continuous sequences)
    - Balanced sharding for consistent batch sizes
    - Episode subsampling via sampling rate
    - Integration with LeRobot data format
    - Support for multi-modal data (video, state, action, language)

    Args:
        dataset_path: Path to LeRobot format dataset directory
        embodiment_tag: Embodiment identifier for cross-embodiment training
        modality_configs: Configuration for each modality (sampling, keys)
        video_backend: Video decoding backend ('torchcodec', 'decord', etc.)
        video_backend_kwargs: Additional arguments for video backend
        shard_size: Target number of chunks per shard
        episode_sampling_rate: Fraction of episode chunks to use (for efficiency)
        chunk_len: Length of each continuous chunk (burn_in + unroll)
        chunk_stride: Stride between chunk windows (default: 1 for dense sampling)
        seed: Random seed for reproducible sharding and sampling
        allow_padding: Whether to allow padding of indices to valid range [0, max_length - 1]

    Example:
        >>> dataset = ShardedChunkDataset(
        ...     dataset_path="/path/to/lerobot_dataset",
        ...     embodiment_tag=EmbodimentTag.LIBERO_PANDA,
        ...     modality_configs={...},
        ...     chunk_len=16,
        ...     chunk_stride=1,
        ...     shard_size=64,
        ...     episode_sampling_rate=0.1,
        ... )
        >>> shard_data = dataset.get_shard(0)  # Get first shard of processed chunks
    """

    def __init__(
        self,
        dataset_path: str | Path,
        embodiment_tag: EmbodimentTag,
        modality_configs: dict[str, ModalityConfig],
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict[str, Any] | None = None,
        shard_size: int = 2**6,  # 64 chunks (smaller than single-step since chunks are larger)
        episode_sampling_rate: float = 0.1,
        chunk_len: int = 16,
        chunk_stride: int = 1,
        seed: int = 42,
        allow_padding: bool = False,
    ):
        """Initialize chunk dataset with sharding configuration."""
        super().__init__(dataset_path)
        self.embodiment_tag = embodiment_tag
        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs
        self.shard_size = shard_size
        self.episode_sampling_rate = episode_sampling_rate
        self.chunk_len = chunk_len
        self.chunk_stride = chunk_stride
        self.seed = seed
        self.allow_padding = allow_padding
        self.processor = None
        self.rng = np.random.default_rng(seed)
        action_delta_indices = modality_configs["action"].delta_indices
        self.action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1

        self.episode_loader = LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
        )

        # Create balanced shards from episode chunks
        self.shard_dataset()

    def get_effective_episode_length(self, episode_index: int) -> int:
        """Get the effective episode length accounting for action horizon and chunk length."""
        original_length = self.episode_loader.get_episode_length(episode_index)
        # Single-step effective length: original_length - action_horizon + 1
        # For chunks, we need to subtract (chunk_len - 1) to ensure the last chunk doesn't overflow
        single_step_effective = max(0, original_length - self.action_horizon + 1)
        chunk_effective = max(0, single_step_effective - (self.chunk_len - 1))
        return chunk_effective

    def get_valid_chunk_starts(self, episode_index: int) -> np.ndarray:
        """Get all valid starting indices for chunks in an episode."""
        effective_length = self.get_effective_episode_length(episode_index)
        if effective_length <= 0:
            return np.array([], dtype=int)
        # Generate chunk start indices with stride
        chunk_starts = np.arange(0, effective_length, self.chunk_stride)
        return chunk_starts

    def shard_dataset(self):
        """
        Create balanced shards by distributing episode chunks across shards.

        The sharding process:
        1. Shuffle episode order for randomization
        2. For each episode, generate all valid chunk start indices
        3. Sample chunks according to episode_sampling_rate
        4. Distribute chunks across shards to balance shard sizes
        5. Use greedy assignment to minimize shard size variance

        This approach ensures:
        - Balanced shard sizes for consistent training batches
        - Diversity within shards (mix of episodes and chunk positions)
        - Reproducible sharding based on seed
        """
        shuffled_episode_indices = self.rng.permutation(len(self.episode_loader.episode_lengths))

        assert len(shuffled_episode_indices) > 0, (
            f"No valid trajectories found for dataset {self.dataset_path}"
        )

        # Collect all valid chunks from all episodes
        all_chunks = []
        for ep_idx in shuffled_episode_indices:
            chunk_starts = self.get_valid_chunk_starts(ep_idx)
            if len(chunk_starts) == 0:
                continue
            # Sample chunks according to episode_sampling_rate
            num_chunks_to_sample = max(1, int(len(chunk_starts) * self.episode_sampling_rate))
            sampled_starts = self.rng.choice(chunk_starts, size=num_chunks_to_sample, replace=False)
            for start_t in sampled_starts:
                all_chunks.append((ep_idx, start_t))

        # Shuffle all chunks for randomization
        self.rng.shuffle(all_chunks)

        # Calculate number of shards
        num_shards = max(1, np.ceil(len(all_chunks) / self.shard_size).astype(int))

        # Initialize shard containers
        sharded_chunks = [[] for _ in range(num_shards)]
        shard_lengths = np.zeros(num_shards, dtype=int)

        # Distribute chunks across shards using greedy assignment
        for ep_idx, start_t in all_chunks:
            # Assign to shard with minimum current length (greedy balancing)
            shard_index = np.argmin(shard_lengths)
            sharded_chunks[shard_index].append((ep_idx, start_t))
            shard_lengths[shard_index] += 1

        # Validate shard creation
        assert all(shard_lengths[i] > 0 for i in range(num_shards)), (
            "All shards must have length greater than 0"
        )

        print(f"Generated {num_shards} shards for dataset {self.dataset_path}")
        print(
            f"Total chunks: {len(all_chunks)}, average shard length: {len(all_chunks) / num_shards:.2f}, "
            f"shard length std: {np.std(shard_lengths):.2f}"
        )
        self.sharded_chunks = sharded_chunks
        self.shard_lengths = shard_lengths

    def __len__(self):
        """Return the number of shards in the dataset."""
        return len(self.shard_lengths)

    def get_chunk_datapoint(self, episode_data: pd.DataFrame, start_t: int) -> dict:
        """
        Extract and process a continuous chunk from episode data.

        Converts raw episode data into a list of VLAStepData structures (one per timestep)
        and applies the configured processor to each step.

        Args:
            episode_data: Complete episode DataFrame from LeRobotEpisodeLoader
            start_t: Starting timestep index within the episode

        Returns:
            Dictionary containing:
                - episode_id: Episode identifier
                - start_t: Starting timestep
                - steps: List of L processed step dictionaries (from processor)
                - chunk_len: Length of the chunk

        Raises:
            AssertionError: If processor is not set before calling this method
        """
        assert self.processor is not None, "Processor must be set before getting datapoints"

        steps = []
        for t_offset in range(self.chunk_len):
            step_t = start_t + t_offset
            vla_step_data = extract_step_data(
                episode_data,
                step_t,
                self.modality_configs,
                self.embodiment_tag,
                self.allow_padding,
            )
            # Apply processor to convert to model inputs
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            step_dict = self.processor(messages)
            steps.append(step_dict)

        return {
            "episode_id": episode_data.name if hasattr(episode_data, "name") else None,
            "start_t": start_t,
            "steps": steps,
            "chunk_len": self.chunk_len,
        }

    def get_shard_length(self, idx: int) -> int:
        """Get the number of chunks in a specific shard."""
        return self.shard_lengths[idx]

    def get_shard(self, idx: int) -> list:
        """
        Load and process all chunks in a specific shard.

        Loads the required episodes and extracts all chunks assigned to this shard,
        applying the configured processor to each step in each chunk.

        Args:
            idx: Shard index to load

        Returns:
            List of processed chunks ready for model training.
            Each chunk is a dict with keys: episode_id, start_t, steps, chunk_len
        """
        chunks = self.sharded_chunks[idx]

        # Group chunks by episode to avoid redundant loading
        from collections import defaultdict

        ep_to_starts: dict[int, list[int]] = defaultdict(list)
        chunk_order: list[tuple[int, int]] = []  # Preserve original order for output
        for ep_idx, start_t in chunks:
            ep_to_starts[ep_idx].append(start_t)
            chunk_order.append((ep_idx, start_t))

        # Pre-load episodes (each episode loaded only once)
        episode_cache: dict[int, pd.DataFrame] = {}
        for ep_idx in ep_to_starts.keys():
            episode_cache[ep_idx] = self.episode_loader[ep_idx]

        # Extract chunks using cached episode data (preserving original order)
        datapoints = []
        for ep_idx, start_t in chunk_order:
            episode_data = episode_cache[ep_idx]
            chunk_datapoint = self.get_chunk_datapoint(episode_data, start_t)
            datapoints.append(chunk_datapoint)

        return datapoints

    def get_dataset_statistics(self) -> dict:
        """Get dataset statistics from the underlying episode loader."""
        return self.episode_loader.get_dataset_statistics()

    def get_initial_actions(self):
        """Get initial actions from the underlying episode loader."""
        return self.episode_loader.get_initial_actions()
