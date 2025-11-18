import os
import sys
import math
import random
import argparse
import asyncio
import datetime
import tempfile
import traceback
import base64
import hashlib
import gc
import time

from typing import Any, Dict, List, Optional, Tuple, cast
from types import SimpleNamespace

from dotenv import load_dotenv
import bittensor as bt
from bittensor.core.chain_data.utils import decode_metadata
from bittensor.core.errors import MetadataError
from substrateinterface import SubstrateInterface
from datasets import load_dataset
from huggingface_hub import list_repo_files
import pandas as pd

# GPU optimizations
try:
    import torch
    if torch.cuda.is_available():
        # Optimize CUDA settings for RTX 4090+
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner for optimal performance
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for better performance
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster matmul on Ampere+
        torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for cuDNN
        # Set memory fraction to allow better memory management
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
        HAS_GPU = True
    else:
        HAS_GPU = False
except ImportError:
    HAS_GPU = False
    torch = None

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from config.config_loader import load_config
from utils import (
    get_sequence_from_protein_code,
    upload_file_to_github,
    get_challenge_params_from_blockhash,
    get_heavy_atom_count,
    # compute_maccs_entropy,  # Not needed when num_molecules=1 (entropy requires multiple molecules)
)
from boltz.wrapper import BoltzWrapper
from btdr import QuicknetBittensorDrandTimelock

# ----------------------------------------------------------------------------
# 1. CONFIG & ARGUMENT PARSING
# ----------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments and merges with config defaults.

    Returns:
        argparse.Namespace: The combined configuration object.
    """
    parser = argparse.ArgumentParser()
    # Add override arguments for network.
    parser.add_argument('--network', default=os.getenv('SUBTENSOR_NETWORK'), help='Network to use')
    # Adds override arguments for netuid.
    parser.add_argument('--netuid', type=int, default=68, help="The chain subnet uid.")
    # Bittensor standard argument additions.
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)

    # Parse combined config
    config = bt.config(parser)

    # Load protein selection params
    config.update(load_config())

    # Final logging dir
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey_str,
            config.netuid,
            'miner',
        )
    )

    # Ensure the logging directory exists.
    os.makedirs(config.full_path, exist_ok=True)
    return config


def load_github_path() -> str:
    """
    Constructs the path for GitHub operations from environment variables.
    
    Returns:
        str: The fully qualified GitHub path (owner/repo/branch/path).
    Raises:
        ValueError: If the final path exceeds 100 characters.
    """
    github_repo_name = os.environ.get('GITHUB_REPO_NAME')  # e.g., "nova"
    github_repo_branch = os.environ.get('GITHUB_REPO_BRANCH')  # e.g., "main"
    github_repo_owner = os.environ.get('GITHUB_REPO_OWNER')  # e.g., "metanova-labs"
    github_repo_path = os.environ.get('GITHUB_REPO_PATH')  # e.g., "data/results" or ""

    if github_repo_name is None or github_repo_branch is None or github_repo_owner is None:
        raise ValueError("Missing one or more GitHub environment variables (GITHUB_REPO_*)")

    if github_repo_path == "":
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}"
    else:
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}/{github_repo_path}"

    if len(github_path) > 100:
        raise ValueError("GitHub path is too long. Please shorten it to 100 characters or less.")

    return github_path


# ----------------------------------------------------------------------------
# 2. LOGGING SETUP
# ----------------------------------------------------------------------------

def setup_logging(config: argparse.Namespace) -> None:
    """
    Sets up Bittensor logging.

    Args:
        config (argparse.Namespace): The miner configuration object.
    """
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running miner for subnet: {config.netuid} on network: {config.subtensor.network} with config:")
    bt.logging.info(config)


# ----------------------------------------------------------------------------
# 3. BITTENSOR & NETWORK SETUP
# ----------------------------------------------------------------------------

async def setup_bittensor_objects(config: argparse.Namespace) -> Tuple[Any, Any, Any, int, int]:
    """
    Initializes wallet, subtensor, and metagraph. Fetches the epoch length
    and calculates the miner UID.

    Args:
        config (argparse.Namespace): The miner configuration object.

    Returns:
        tuple: A 5-element tuple of
            (wallet, subtensor, metagraph, miner_uid, epoch_length).
    """
    bt.logging.info("Setting up Bittensor objects.")

    # Initialize wallet
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    # Initialize subtensor (asynchronously)
    # Note: Don't use context manager here - we need to keep the connection alive
    try:
        subtensor = bt.async_subtensor(network=config.network)
        await subtensor.initialize()
        bt.logging.info(f"Connected to subtensor network: {config.network}")
        
        # Sync metagraph
        metagraph = await subtensor.metagraph(config.netuid)
        await metagraph.sync()
        bt.logging.info(f"Metagraph synced successfully.")

        bt.logging.info(f"Subtensor: {subtensor}")
        bt.logging.info(f"Metagraph synced: {metagraph}")

        # Get miner UID
        try:
            miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
            bt.logging.info(f"Miner UID: {miner_uid}")
        except ValueError:
            bt.logging.error(f"Hotkey {wallet.hotkey.ss58_address} not found in metagraph. Are you registered?")
            raise ValueError(f"Hotkey not registered on subnet {config.netuid}")

        # Query epoch length
        node = SubstrateInterface(url=config.network)
        # Set epoch_length to tempo + 1
        epoch_length = node.query("SubtensorModule", "Tempo", [config.netuid]).value + 1
        bt.logging.info(f"Epoch length query successful: {epoch_length} blocks")

        return wallet, subtensor, metagraph, miner_uid, epoch_length
    except Exception as e:
        bt.logging.error(f"Failed to setup Bittensor objects: {e}")
        bt.logging.error("Please check your network connection and the subtensor network status")
        raise

# ----------------------------------------------------------------------------
# 4. GPU MEMORY MANAGEMENT
# ----------------------------------------------------------------------------

def clear_gpu_memory():
    """Clear GPU memory cache and run garbage collection."""
    if HAS_GPU and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


def get_gpu_memory_info() -> Optional[Dict[str, float]]:
    """Get current GPU memory usage information."""
    if HAS_GPU and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated
        }
    return None


# ----------------------------------------------------------------------------
# 5. DATA SETUP
# ----------------------------------------------------------------------------

def stream_random_chunk_from_dataset(dataset_repo: str, chunk_size: int) -> Any:
    """
    Streams a random chunk from the specified Hugging Face dataset repo.

    Args:
        dataset_repo (str): Hugging Face dataset repository path (user/repo).
        chunk_size (int): Size of each chunk to stream.

    Returns:
        Any: A batched (chunked) dataset iterator.
    
    Raises:
        ValueError: If no CSV files are found in the dataset repository.
    """
    files = list_repo_files(dataset_repo, repo_type='dataset')
    files = [file for file in files if file.endswith('.csv')]
    
    if not files:
        raise ValueError(f"No CSV files found in dataset repository: {dataset_repo}")
    
    random_file = random.choice(files)

    dataset_dict = load_dataset(
        dataset_repo,
        data_files={'train': random_file},
        streaming=True,
    )
    dataset = dataset_dict['train']
    batched = dataset.batch(chunk_size)
    return batched


# ----------------------------------------------------------------------------
# 6. INFERENCE AND SUBMISSION LOGIC
# ----------------------------------------------------------------------------

async def run_boltz_model_loop(state: Dict[str, Any]) -> None:
    """
    Continuously runs the Boltz model on batches of molecules from Hugging Face dataset.
    Updates the best candidate whenever a higher score is found, but only submits when close to epoch end.

    Args:
        state (dict): A shared state dict containing references to:
            'chunk_size', 'hugging_face_dataset_repo', 'boltz', 'best_score',
            'candidate_product', 'submission_interval', 'last_submission_time',
            'last_submitted_product', 'shutdown_event', etc.
    """
    bt.logging.info("Starting Boltz model inference loop.")
    if HAS_GPU:
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            bt.logging.info(f"GPU Memory - Allocated: {gpu_info['allocated_gb']:.2f} GB, "
                          f"Reserved: {gpu_info['reserved_gb']:.2f} GB")
    
    # Cache block hash to reduce API calls (update every 10 iterations)
    cached_block = None
    cached_block_hash = None
    block_hash_cache_ttl = 10
    iteration_count = 0
    last_gpu_cleanup = time.time()
    gpu_cleanup_interval = 300  # Clean GPU memory every 5 minutes

    # Create dataset iterator once at the start
    dataset_iter = stream_random_chunk_from_dataset(
        dataset_repo=state['hugging_face_dataset_repo'],
        chunk_size=state['chunk_size']
    )

    while not state['shutdown_event'].is_set():
        try:
            for chunk in dataset_iter:
                if state['shutdown_event'].is_set():
                    break

                iteration_count += 1
                
                # Periodic GPU memory cleanup
                current_time = time.time()
                if HAS_GPU and (current_time - last_gpu_cleanup) > gpu_cleanup_interval:
                    clear_gpu_memory()
                    last_gpu_cleanup = current_time
                    gpu_info = get_gpu_memory_info()
                    if gpu_info:
                        bt.logging.debug(f"GPU Memory after cleanup - Allocated: {gpu_info['allocated_gb']:.2f} GB")

                df = pd.DataFrame.from_dict(chunk)
                
                # Check for required columns
                if 'product_smiles' not in df.columns or 'product_name' not in df.columns:
                    bt.logging.warning(f"Chunk missing required columns. Available: {df.columns.tolist()}")
                    del df
                    continue
                
                # Optimized data cleaning with vectorized operations
                df['product_name'] = df['product_name'].str.replace('"', '', regex=False)
                df['product_smiles'] = df['product_smiles'].str.replace('"', '', regex=False)

                # Vectorized heavy atom counting (more efficient than apply)
                try:
                    df['heavy_atoms'] = df['product_smiles'].apply(get_heavy_atom_count)
                    df = df[df['heavy_atoms'] >= state['config'].min_heavy_atoms].copy()
                except Exception as e:
                    bt.logging.warning(f"Error calculating heavy atoms: {e}")
                    del df
                    continue
                
                if df.empty or len(df) < state['config'].num_molecules:
                    del df  # Explicit cleanup
                    continue

                # Cache block hash to reduce API calls
                if cached_block is None or iteration_count % block_hash_cache_ttl == 0:
                    current_block = await state['subtensor'].get_current_block()
                    final_block_hash = await state['subtensor'].determine_block_hash(current_block)
                    cached_block = current_block
                    cached_block_hash = final_block_hash
                else:
                    current_block = cached_block
                    final_block_hash = cached_block_hash

                # Prepare molecules for Boltz scoring
                # Format: valid_molecules_by_uid = {uid: {"smiles": [...], "names": [...]}}
                # For miner, we use uid=0 as a placeholder
                # Filter out NaN values from dataframe to ensure alignment
                df = df.dropna(subset=['product_smiles', 'product_name']).copy()
                
                if df.empty:
                    bt.logging.debug("No valid SMILES or names after filtering NaN values")
                    continue
                
                valid_molecules_by_uid = {
                    0: {
                        "smiles": df['product_smiles'].tolist(),
                        "names": df['product_name'].tolist()
                    }
                }

                # Initialize score_dict for Boltz
                score_dict = {
                    0: {
                        # "entropy": None,  # Not needed when num_molecules=1
                        "entropy_boltz": None,  # Still needed by Boltz wrapper
                        "block_submitted": None,
                        "push_time": "",
                        "boltz_score": None
                    }
                }

                # Score molecules with Boltz
                # Convert config namespace to dict for Boltz (which expects dict-style access)
                # argparse.Namespace can be converted to dict using vars()
                if isinstance(state['config'], dict):
                    config_dict = state['config']
                else:
                    config_dict = vars(state['config'])
                try:
                    # Clear GPU cache before scoring to ensure maximum available memory
                    if HAS_GPU and iteration_count % 5 == 0:  # Every 5 iterations
                        clear_gpu_memory()
                    
                    state['boltz'].score_molecules_target(
                        valid_molecules_by_uid=valid_molecules_by_uid,
                        score_dict=score_dict,
                        subnet_config=config_dict,
                        final_block_hash=final_block_hash
                    )
                    
                    # Clear GPU cache after scoring to free up memory
                    if HAS_GPU:
                        clear_gpu_memory()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        bt.logging.warning(f"GPU out of memory, clearing cache and retrying...")
                        clear_gpu_memory()
                        # Retry once after clearing memory
                        try:
                            state['boltz'].score_molecules_target(
                                valid_molecules_by_uid=valid_molecules_by_uid,
                                score_dict=score_dict,
                                subnet_config=config_dict,
                                final_block_hash=final_block_hash
                            )
                            clear_gpu_memory()
                        except Exception as e2:
                            bt.logging.error(f"Error scoring molecules with Boltz after retry: {e2}")
                            continue
                    else:
                        bt.logging.error(f"Error scoring molecules with Boltz: {e}")
                        traceback.print_exc()
                        continue
                except Exception as e:
                    bt.logging.error(f"Error scoring molecules with Boltz: {e}")
                    traceback.print_exc()
                    continue

                # Get Boltz scores
                boltz_score = score_dict[0].get('boltz_score')
                if boltz_score is None or boltz_score == -math.inf:
                    bt.logging.debug("No valid Boltz scores returned")
                    continue

                # Create a dataframe with scores for sorting
                # Since Boltz returns a single score per UID (average), we need to work with the per-molecule scores
                # Check if per_molecule_metric is available
                if hasattr(state['boltz'], 'per_molecule_metric') and 0 in state['boltz'].per_molecule_metric:
                    per_mol_scores = state['boltz'].per_molecule_metric[0]
                    # Use vectorized map for better performance
                    df['boltz_score'] = df['product_smiles'].map(per_mol_scores).fillna(-math.inf)
                else:
                    # Fallback: use average score for all molecules
                    df['boltz_score'] = boltz_score

                # Sort by Boltz score
                df.sort_values(by=['boltz_score'], ascending=[False], inplace=True)
                df.reset_index(drop=True, inplace=True)

                # Select top molecules (up to num_molecules)
                num_top = min(state['config'].num_molecules, len(df))
                top_molecules = df.iloc[:num_top]
                
                if not top_molecules.empty:
                    # entropy = compute_maccs_entropy(top_molecules['product_smiles'].tolist())  # Not needed when num_molecules=1
                    scores_sum = top_molecules['boltz_score'].sum()
                    
                    # Check for NaN or invalid scores
                    if pd.isna(scores_sum) or not math.isfinite(scores_sum):
                        bt.logging.debug("Invalid score sum (NaN or inf), skipping...")
                        del df
                        del valid_molecules_by_uid
                        del score_dict
                        continue
                    
                    # Calculate final score (entropy bonus disabled when num_molecules=1)
                    # When num_molecules=1, entropy has no meaning (requires multiple molecules for diversity)
                    # entropy_weight = getattr(state['config'], 'entropy_start_weight', 0.0)
                    # if scores_sum > getattr(state['config'], 'entropy_bonus_threshold', 0.0):
                    #     final_score = scores_sum * (entropy_weight + entropy)
                    # else:
                    final_score = scores_sum

                    if final_score > state['best_score']:
                        state['best_score'] = final_score
                        state['candidate_product'] = ','.join(top_molecules['product_name'].tolist())
                        bt.logging.info(f"New best score: {state['best_score']}, Candidates: {state['candidate_product']}")

                    # Check if we're close to epoch end (20 blocks away)
                    next_epoch_block = ((current_block // state['epoch_length']) + 1) * state['epoch_length']
                    blocks_until_epoch = next_epoch_block - current_block
                    
                    bt.logging.debug(f"Current block: {current_block}, Epoch length: {state['epoch_length']}, Next epoch block: {next_epoch_block}, Blocks until epoch: {blocks_until_epoch}")
                    
                    if state['candidate_product'] and blocks_until_epoch <= 20:
                        bt.logging.info(f"Close to epoch end ({blocks_until_epoch} blocks remaining), attempting submission...")
                        if state['candidate_product'] != state['last_submitted_product']:
                            bt.logging.info("Attempting to submit new candidate...")
                            try:
                                await submit_response(state)
                            except Exception as e:
                                bt.logging.error(f"Error submitting response: {e}")
                        else:
                            bt.logging.info("Skipping submission - same product as last submission")

                # Clean up dataframe to free memory
                del df
                del valid_molecules_by_uid
                del score_dict
                
                # Small sleep to allow other async operations
                await asyncio.sleep(1)  # Reduced from 2 to 1 for faster processing
            
            # If we exit the for loop, the iterator is exhausted - recreate it
            bt.logging.debug("Dataset iterator exhausted, recreating...")
            try:
                dataset_iter = stream_random_chunk_from_dataset(
                    dataset_repo=state['hugging_face_dataset_repo'],
                    chunk_size=state['chunk_size']
                )
            except Exception as e:
                bt.logging.error(f"Error recreating dataset iterator: {e}")
                await asyncio.sleep(10)  # Wait longer before retrying
                continue
            await asyncio.sleep(2)  # Brief pause before recreating iterator

        except StopIteration:
            # Iterator exhausted, recreate it
            bt.logging.debug("Dataset iterator exhausted (StopIteration), recreating...")
            try:
                dataset_iter = stream_random_chunk_from_dataset(
                    dataset_repo=state['hugging_face_dataset_repo'],
                    chunk_size=state['chunk_size']
                )
            except Exception as e:
                bt.logging.error(f"Error recreating dataset iterator: {e}")
                await asyncio.sleep(10)  # Wait longer before retrying
                continue
            await asyncio.sleep(2)
        except Exception as e:
            bt.logging.error(f"Error in Boltz model loop: {e}")
            traceback.print_exc()
            # Don't set shutdown_event on general errors, just log and continue
            await asyncio.sleep(5)  # Wait before retrying


async def submit_response(state: Dict[str, Any]) -> None:
    """
    Encrypts and submits the current candidate product as a chain commitment and uploads
    the encrypted response to GitHub. If the chain accepts the commitment, we finalize it.

    Args:
        state (dict): Shared state dictionary containing references to:
            'bdt', 'miner_uid', 'candidate_product', 'subtensor', 'wallet', 'config',
            'github_path', etc.
    """
    candidate_product = state['candidate_product']
    if not candidate_product:
        bt.logging.warning("No candidate product to submit")
        return

    bt.logging.info(f"Starting submission process for product: {candidate_product}")
    
    # 1) Encrypt the response
    current_block = await state['subtensor'].get_current_block()
    encrypted_response = state['bdt'].encrypt(state['miner_uid'], candidate_product, current_block)
    bt.logging.info(f"Encrypted response generated successfully")

    # 2) Create temp file, write content
    tmp_file = tempfile.NamedTemporaryFile(delete=True)
    encoded_content = None
    filename = None
    commit_content = None
    
    try:
        with open(tmp_file.name, 'w+') as f:
            f.write(str(encrypted_response))
            f.flush()

            # Read, base64-encode
            f.seek(0)
            content_str = f.read()
            encoded_content = base64.b64encode(content_str.encode()).decode()

            # Generate short hash-based filename
            filename = hashlib.sha256(content_str.encode()).hexdigest()[:20]
            commit_content = f"{state['github_path']}/{filename}.txt"
            bt.logging.info(f"Prepared commit content: {commit_content}")
    except Exception as e:
        bt.logging.error(f"Error preparing file for submission: {e}")
        return

    # 3) Attempt chain commitment
    if not commit_content:
        bt.logging.error("Failed to prepare commit content")
        return
        
    bt.logging.info(f"Attempting chain commitment...")
    try: 
        commitment_status = await state['subtensor'].set_commitment(
            wallet=state['wallet'],
            netuid=state['config'].netuid,
            data=commit_content
        )
        bt.logging.info(f"Chain commitment status: {commitment_status}")
    except MetadataError:
        bt.logging.info("Too soon to commit again. Will keep looking for better candidates.")
        return
    except Exception as e:
        bt.logging.error(f"Error setting commitment: {e}")
        return

    # 4) If chain commitment success, upload to GitHub
    if commitment_status and encoded_content and filename:
        try:
            bt.logging.info(f"Commitment set successfully for {commit_content}")
            bt.logging.info("Attempting GitHub upload...")
            github_status = upload_file_to_github(filename, encoded_content)
            if github_status:
                bt.logging.info(f"File uploaded successfully to {commit_content}")
                state['last_submitted_product'] = candidate_product
                state['last_submission_time'] = datetime.datetime.now()
            else:
                bt.logging.error(f"Failed to upload file to GitHub for {commit_content}")
        except Exception as e:
            bt.logging.error(f"Failed to upload file for {commit_content}: {e}")


# ----------------------------------------------------------------------------
# 6. MAIN MINING LOOP
# ----------------------------------------------------------------------------

async def run_miner(config: argparse.Namespace) -> None:
    """
    The main mining loop, orchestrating:
      - Bittensor objects initialization
      - Model initialization
      - Fetching new proteins each epoch
      - Running inference and submissions
      - Periodically syncing metagraph

    Args:
        config (argparse.Namespace): The miner configuration object.
    """

    # 1) Setup wallet, subtensor, metagraph, etc.
    wallet, subtensor, metagraph, miner_uid, epoch_length = await setup_bittensor_objects(config)

    # 2) Prepare shared state
    # Adaptive chunk size based on GPU availability (larger chunks for RTX 4090+)
    base_chunk_size = 128
    if HAS_GPU:
        # Increase chunk size for high-end GPUs to better utilize GPU memory
        adaptive_chunk_size = base_chunk_size * 2  # 256 for GPU
        bt.logging.info(f"GPU detected - using optimized chunk size: {adaptive_chunk_size}")
    else:
        adaptive_chunk_size = base_chunk_size
        bt.logging.info(f"No GPU detected - using base chunk size: {adaptive_chunk_size}")
    
    state: Dict[str, Any] = {
        # environment / config
        'config': config,
        'hugging_face_dataset_repo': 'Metanova/SAVI-2020',
        'chunk_size': adaptive_chunk_size,
        'submission_interval': 1200,

        # GitHub
        'github_path': load_github_path(),

        # Bittensor
        'wallet': wallet,
        'subtensor': subtensor,
        'metagraph': metagraph,
        'miner_uid': miner_uid,
        'epoch_length': epoch_length,

        # Models - Boltz model (single instance)
        'boltz': BoltzWrapper(),
        'bdt': QuicknetBittensorDrandTimelock(),

        # Inference state
        'candidate_product': None,
        'best_score': float('-inf'),
        'last_submitted_product': None,
        'last_submission_time': None,
        'shutdown_event': asyncio.Event(),

        # Challenges (kept for reference, but Boltz only uses weekly_target from config)
        'current_challenge_targets': [],
        'last_challenge_targets': [],
        'current_challenge_antitargets': [],
        'last_challenge_antitargets': [],
    }
    
    # Log GPU information at startup
    if HAS_GPU:
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            bt.logging.info(f"GPU Memory Info - Allocated: {gpu_info['allocated_gb']:.2f} GB, "
                          f"Reserved: {gpu_info['reserved_gb']:.2f} GB, "
                          f"Max Allocated: {gpu_info['max_allocated_gb']:.2f} GB")

    bt.logging.info("Entering main miner loop...")

    # 3) If we start mid-epoch, obtain most recent proteins from block hash
    current_block = await subtensor.get_current_block()
    last_boundary = (current_block // epoch_length) * epoch_length
    next_boundary = last_boundary + epoch_length

    # If we start too close to epoch end, wait for next epoch
    if next_boundary - current_block < 20:
        bt.logging.info(f"Too close to epoch end, waiting for next epoch to start...")
        block_to_check = next_boundary
        await asyncio.sleep(12*10)
    else:
        block_to_check = last_boundary

    block_hash = await subtensor.determine_block_hash(block_to_check)
    startup_proteins = get_challenge_params_from_blockhash(
        block_hash=block_hash,
        weekly_target=config.weekly_target,
        num_antitargets=config.num_antitargets
    )

    if startup_proteins:
        state['current_challenge_targets'] = startup_proteins["targets"]
        state['last_challenge_targets'] = startup_proteins["targets"]
        state['current_challenge_antitargets'] = startup_proteins["antitargets"]
        state['last_challenge_antitargets'] = startup_proteins["antitargets"]
        bt.logging.info(f"Startup targets: {startup_proteins['targets']}, antitargets: {startup_proteins['antitargets']}")
        bt.logging.info(f"Boltz model initialized and ready (using weekly_target: {config.weekly_target})")

        # 4) Launch the inference loop
        try:
            state['inference_task'] = asyncio.create_task(run_boltz_model_loop(state))
            bt.logging.debug("Boltz inference started.")
        except Exception as e:
            bt.logging.error(f"Error starting inference: {e}")

    # 5) Main epoch-based loop
    while True:
        try:
            current_block = await subtensor.get_current_block()

            # If we are at an epoch boundary, fetch new proteins
            if current_block % epoch_length == 0:
                bt.logging.info(f"Found epoch boundary at block {current_block}.")
                
                block_hash = await subtensor.determine_block_hash(current_block)
                
                new_proteins = get_challenge_params_from_blockhash(
                    block_hash=block_hash,
                    weekly_target=config.weekly_target,
                    num_antitargets=config.num_antitargets
                )
                if (new_proteins and 
                    (new_proteins["targets"] != state['last_challenge_targets'] or 
                     new_proteins["antitargets"] != state['last_challenge_antitargets'])):
                    state['current_challenge_targets'] = new_proteins["targets"]
                    state['last_challenge_targets'] = new_proteins["targets"]
                    state['current_challenge_antitargets'] = new_proteins["antitargets"]
                    state['last_challenge_antitargets'] = new_proteins["antitargets"]
                    bt.logging.info(f"New proteins - targets: {new_proteins['targets']}, antitargets: {new_proteins['antitargets']}")

                # Cancel old inference, reset relevant state
                if 'inference_task' in state and state['inference_task']:
                    if not state['inference_task'].done():
                        state['shutdown_event'].set()
                        bt.logging.debug("Shutdown event set for old inference task.")
                        await state['inference_task']

                # Reset best score and candidate
                state['candidate_product'] = None
                state['best_score'] = float('-inf')
                state['last_submitted_product'] = None
                state['shutdown_event'] = asyncio.Event()

                # Boltz model is already initialized and uses weekly_target from config
                bt.logging.info(f"Boltz model ready for new epoch (using weekly_target: {config.weekly_target})")

                # Start new inference
                try:
                    state['inference_task'] = asyncio.create_task(run_boltz_model_loop(state))
                    bt.logging.debug("New Boltz inference task started.")
                except Exception as e:
                    bt.logging.error(f"Error starting new inference: {e}")

            # Periodically update our knowledge of the network
            if current_block % 60 == 0:
                await metagraph.sync()
                log = (
                    f"Block: {metagraph.block.item()} | "
                    f"Number of nodes: {metagraph.n} | "
                    f"Current epoch: {metagraph.block.item() // epoch_length}"
                )
                bt.logging.info(log)

            await asyncio.sleep(1)

        except RuntimeError as e:
            bt.logging.error(e)
            traceback.print_exc()

        except KeyboardInterrupt:
            bt.logging.success("Keyboard interrupt detected. Exiting miner.")
            break


# ----------------------------------------------------------------------------
# 7. ENTRY POINT
# ----------------------------------------------------------------------------

async def main() -> None:
    """
    Main entry point for asynchronous execution of the miner logic.
    """
    config = parse_arguments()
    setup_logging(config)
    await run_miner(config)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())