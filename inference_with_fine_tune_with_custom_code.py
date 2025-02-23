### INFO: This is a helper script to allow participants to confirm their model is working!
import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraModel, LoraConfig

from fsdp_utils import AppState

from typing import List, Callable, Optional
import re
from pydantic import BaseModel, Field, TypeAdapter
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool
import requests
import json

adapter_name = "ExampleLora"

# INFO: This is a helper to map model names to StrongCompute Dataset ID's which store their weights!
model_weight_ids = {
    "DeepSeek-R1-Distill-Llama-70B": "e4b2dc79-79af-4a80-be71-c509469449b4",
    "DeepSeek-R1-Distill-Llama-8B": "38b32289-7d34-4c72-9546-9d480f676840",
    "DeepSeek-R1-Distill-Qwen-1.5B": "6c796efa-7063-4a74-99b8-aab1c728ad98",
    "DeepSeek-R1-Distill-Qwen-14B": "39387beb-9824-4629-b19b-8f7b8f127150",
    "DeepSeek-R1-Distill-Qwen-32B": "84c2b2cb-95b4-4ce6-a2d4-6f210afad36b",
    "DeepSeek-R1-Distill-Qwen-7B": "a792646c-39f5-4971-a169-425324fec87b",
}

# TODO: set this to the model you chose from the dropdown at container startup!
MODEL_NAME_SETME = "DeepSeek-R1-Distill-Qwen-1.5B"
mounted_dataset_path = f"/data/{model_weight_ids[MODEL_NAME_SETME]}"

# INFO: Loads the model WEIGHTS (assuming you've mounted it to your container!)
tokenizer = AutoTokenizer.from_pretrained(mounted_dataset_path)
model = AutoModelForCausalLM.from_pretrained(
    mounted_dataset_path, 
    use_cache=False, 
    torch_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0, # set to zero to see identical loss on all ranks
)

model = LoraModel(model, lora_config, adapter_name).to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
state_dict = { "app": AppState(model, optimizer)}
dcp.load(state_dict=state_dict, checkpoint_id="/shared/artifacts/<experiment-id>/checkpoints/CHKxx") ## UPDATE WITH PATH TO CHECKPOINT DIRECTORY

prompt = "Can you tell me the current weather in Sydney?"

class ToolCall(BaseModel):
    tool: str = Field(..., description="Name of the tool to call")
    args: dict = Field(..., description="Arguments to pass to the tool")

def process_user_input(prompt: str, tools: List[Callable]) -> str:
    # https://arxiv.org/abs/2501.12948
    deepseek_r1_input = f'''
    A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
    The assistant first thinks about the reasoning process in the mind and then provides the user
    with the answer. The reasoning process and answer are enclosed within <think> </think> and
    <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
    <answer> answer here </answer>. User: {prompt}. Assistant:'''

    encoding = tokenizer(deepseek_r1_input, return_tensors="pt")

    input_ids = encoding['input_ids'].to("cuda")
    attention_mask = encoding['attention_mask'].to("cuda")

    generate_ids = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, max_new_tokens=100, do_sample=True, temperature=0.8)
    answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # print(answer[0])
    last_response = answer[0]

    # Create JSON parser
    # json_parser = JsonOutputParser(pydantic_object=ToolCall)

    # Process JSON response
    answer_match = re.search(r'<answer>\s*(\[.*?\])\s*</answer>', last_response, re.DOTALL)

    if answer_match:
        answer_content = answer_match.group(1).strip()
        
        try:
            # Parse JSON directly
            json_data = json.loads(answer_content)
            # Get the first item if it's a list
            tool_data = json_data[0] if isinstance(json_data, list) else json_data
            
            # Create ToolCall instance
            tool_call = ToolCall(**tool_data)

            tool_dict = {tool.name: tool for tool in tools}

            if tool_call.tool in tool_dict:
                result = tool_dict[tool_call.tool].invoke(tool_call.args)
                return result
            else:
                return "Error: Unknown tool"
        except Exception as e:
            error_msg = f"Error processing tool call: {str(e)}"
            print(error_msg)
            return error_msg
    else:
        return last_response

# Example tool definitions
@tool
def get_weather(location: str, api_key: Optional[str] = None) -> str:
    """Get the current weather for a specific location.
    
    Args:
        location: The city name or location to get weather for
        api_key: Optional OpenWeatherMap API key. If not provided, will use environment variable.
    
    Returns:
        str: A string containing the weather information
    """
    if api_key is None:
        api_key = "889fccd150635f5623de7326aed6a9ca"
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    try:
        params = {
            "q": location,
            "appid": api_key,
            "units": "metric"
        }
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        weather_data = response.json()
        
        temperature = weather_data["main"]["temp"]
        description = weather_data["weather"][0]["description"]
        humidity = weather_data["main"]["humidity"]
        
        weather_info = (
            f"Current weather in {location}:\n"
            f"Temperature: {temperature}°C\n"
            f"Conditions: {description}\n"
            f"Humidity: {humidity}%"
        )
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"

# Run the process
result_tool = process_user_input(prompt, [get_weather])
print(result_tool)