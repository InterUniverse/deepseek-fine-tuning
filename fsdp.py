import os
from dotenv import load_dotenv
import functools
import logging
import warnings
import json

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraModel, LoraConfig
from huggingface_hub import login
from datasets import load_dataset

from cycling_utils import atomic_torch_save, AtomicDirectory, TimestampedTimer, InterruptableDistributedSampler

import argparse
from fsdp_utils import bfSixteen_ready, bfSixteen_policy, count_trainable_parameters, AppState, get_args_parser

# Hardcoded token (VERY BAD PRACTICE - ONLY FOR EXTREME SHORT-TERM POC)
HF_TOKEN = "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"  # Replace with your actual token

login(token=HF_TOKEN)  # Log in immediately

# load_dotenv()  # Load environment variables from .env
# hf_token = os.environ.get("HF_TOKEN") 
# if hf_token:
#     login(token=hf_token)
# else:
#     raise ValueError("HF_TOKEN not found in .env file")

timer = TimestampedTimer("Start")

# suppressing warnings about missing modules in state_dict
logger = logging.getLogger("torch.distributed.fsdp._state_dict_utils")
logger.setLevel(logging.ERROR)
# suppress warnings about "UserWarning: `_get_pg_default_device` will be deprecated" while saving and loading
warnings.filterwarnings("ignore", category=UserWarning)

ADAPTER_NAME = "ExampleLora"
SHARD_STRATEGY = ShardingStrategy.FULL_SHARD

def build_prompt_and_response(sample):
    """
    Given a single sample from the xlam-function-calling-60k dataset,
    return (input_text, target_text) strings suitable for training.
    """
    try:
        # Convert sample to dict if it's not already
        sample_dict = dict(sample)
        
        query = sample_dict["query"]
        tools = sample_dict["tools"]
        answers = sample_dict["answers"]

        # Ensure tools is a list of dictionaries
        if isinstance(tools, str):
            tools = json.loads(tools)
        
        # Get tool names
        tool_names = [t["name"] if isinstance(t, dict) else t for t in tools]
        tools_str = ", ".join(tool_names) if tool_names else "No tools listed"

        input_text = (
            "You are a helpful reasoning model that can call tools by returning JSON. "
            "Below is the user's query. You can use these tools:\n"
            f"{tools_str}\n\n"
            f"User Query: {query}\n"
            "Return the correct function call(s) in valid JSON. If multiple calls are needed, "
            "return them as a JSON array. Do not include extraneous text outside the JSON.\n"
        )

        chain_of_thought = (
            "<think>Step-by-step reasoning about how to handle the user's query.</think>"
        )

        # Ensure answers is in the correct format for JSON serialization
        if isinstance(answers, str):
            answers = json.loads(answers)

        answers_json_str = json.dumps(answers, ensure_ascii=False)
        target_text = f"{chain_of_thought}\n\n{answers_json_str}"

        return input_text, target_text
    
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None, None

class FunctionCallingDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        for sample in dataset:
            input_text, target_text = build_prompt_and_response(sample)
            if input_text is not None and target_text is not None:
                self.samples.append({
                    "input_text": input_text,
                    "target_text": target_text
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Combine input and target for causal LM training
        full_text = f"{sample['input_text']}{sample['target_text']}"
        
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels (shift input_ids right)
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

if __name__ == "__main__":
    parser = get_args_parser() # Get the parser object
    args = parser.parse_args()  # Parse the arguments using the parser object

    rank = int(os.environ["RANK"]) # Global rank
    local_device = int(os.environ["LOCAL_RANK"]) # Rank on local node
    world_size = int(os.environ["WORLD_SIZE"]) # Total number of global ranks
    model_path = os.path.join("/data", args.dataset_id)
    torch.cuda.set_device(local_device)

    timer.report(f"Init process group for world size: {world_size}")

    device_mesh = init_device_mesh("cuda", (world_size,))
    saving_group = device_mesh.get_group()
    assert bfSixteen_ready(), "ERROR: System not BF16 ready."

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if rank == 0:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            use_cache=False, 
            torch_dtype=torch.bfloat16
        )
        print(f"Main rank {rank} model params on device: {set([p.data.device for p in model.parameters()])}")
    else:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                use_cache=False, 
                torch_dtype=torch.bfloat16
            )
            print(f"Non-main rank {rank} model params on device: {set([p.data.device for p in model.parameters()])}")

    timer.report(f"Loaded model: {count_trainable_parameters(model)}")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
    )

    model = LoraModel(model, lora_config, ADAPTER_NAME)

    timer.report(f"PEFT model: {count_trainable_parameters(model)}")

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1_000
    )

    model = FSDP(model, 
        auto_wrap_policy=my_auto_wrap_policy,
        sharding_strategy=SHARD_STRATEGY,
        mixed_precision=bfSixteen_policy,
        cpu_offload=CPUOffload(offload_params=True),
        device_id=torch.cuda.current_device(),
        param_init_fn=lambda mod: mod.to_empty(device=torch.cuda.current_device(), recurse=False),
        sync_module_states=True,
        device_mesh=device_mesh
    )

    timer.report("FSDP wrapped model and broadcast to GPUs")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Load and prepare the dataset
    raw_dataset = load_dataset(
    "Salesforce/xlam-function-calling-60k",
    token=HF_TOKEN,
    # token=os.environ['HF_TOKEN'],
    verification_mode="no_checks",
    data_files="xlam_function_calling_60k.json",
    streaming=True
    )

    train_dataset = raw_dataset["train"].shard(num_shards=world_size, index=rank) # Sharding

    dataset = FunctionCallingDataset(train_dataset, tokenizer)

    train_sampler = InterruptableDistributedSampler(dataset)

    batch_size = 2
    dataloader = DataLoader(
        dataset,  # Use the dataset created on each rank
        batch_size=batch_size,
        collate_fn=lambda x: {
            'input_ids': torch.stack([s['input_ids'] for s in x]),
            'attention_mask': torch.stack([s['attention_mask'] for s in x]),
            'labels': torch.stack([s['labels'] for s in x])
        },
        sampler=train_sampler
    )

    # load checkpoint if found
    saver = AtomicDirectory(output_directory=args.chk_path, is_master=rank==0)
    latest_sym = os.path.join(args.chk_path, saver.symlink_name)
    if os.path.exists(latest_sym):
        latest_path = os.readlink(latest_sym)
        state_dict = { "app": AppState(model, optimizer)}
        dcp.load(state_dict=state_dict, checkpoint_id=latest_path)

        train_state = torch.load(os.path.join(latest_path, "train_state.pt"))
        dataloader.sampler.load_state_dict(train_state["sampler"])

        timer.report("Loaded checkpoint")

    # training
    num_epochs = 5
    save_every = 2
    model.train()

    for epoch in range(dataloader.sampler.epoch, num_epochs):
        dataloader.sampler.set_epoch(epoch)

        for step, batch in enumerate(dataloader):
            is_save_step = (step + 1) % save_every == 0
            if is_save_step:
                checkpoint_directory = saver.prepare_checkpoint_directory()
                timer.report("Prepared checkpoint directory")

            input_ids = batch["input_ids"].to(torch.cuda.current_device())
            attention_mask = batch["attention_mask"].to(torch.cuda.current_device())
            labels = batch["labels"].to(torch.cuda.current_device())

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            dataloader.sampler.advance(len(input_ids))
            optimizer.zero_grad()

            timer.report(f"Step {step} Loss: {loss.item()}")

            if is_save_step:
                state_dict = { "app": AppState(model, optimizer) }
                dcp.save(state_dict=state_dict, checkpoint_id=checkpoint_directory, process_group=saving_group)
                torch.save({
                    "sampler": dataloader.sampler.state_dict()
                }, os.path.join(checkpoint_directory, "train_state.pt"))

                saver.atomic_symlink(checkpoint_directory)
                timer.report("Saved checkpoint")

        dataloader.sampler.reset_progress()

    timer.report("Done.")

    dist.barrier()
    dist.destroy_process_group()
