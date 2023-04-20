import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
import argparse
import json, os
from rouge import Rouge
parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default="/root/autodl-fs/Chinese-LLaMA-Alpaca-main/alpaca_combined_hf", type=str)
parser.add_argument('--lora_model', default="/root/autodl-fs/Chinese-LLaMA-Alpaca-main/output", type=str,
                    help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path', default="/root/autodl-fs/Chinese-LLaMA-Alpaca-main/scripts/merged_tokenizer_hf",
                    type=str)
parser.add_argument('--data_file', default="/root/autodl-fs/Chinese-LLaMA-Alpaca-main/data/enoch_fine_tune_dev.json",
                    type=str, help="file that contains instructions (one instruction per line).")
parser.add_argument('--with_prompt', default=True, action='store_true')
parser.add_argument('--interactive', default=False, action='store_true')
parser.add_argument('--predictions_file', default='/root/autodl-fs/Chinese-LLaMA-Alpaca-main/output/predictions.json',
                    type=str)
args = parser.parse_args()

generation_config = dict(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.3,
    max_new_tokens=400
)

# The prompt template below is taken from llama.cpp
# and is slightly different from the one used in training.
# But we find it gives better results
prompt_input = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
)

# sample_data = ["为什么要减少污染，保护环境？"]
instruction = "你现在是一个命名实体识别模型，请你帮我抽取出命名实体识别类别为'安装动作','操作部件位置1','方位','目标部件位置1','操作程序选项','一般动作','操作部件位置2','目标部件2','物理量','目标部件位置2','量词','工作区域','拆卸动作','操作程序','目标部件1','操作部件2','操作部件1','一般工具'的二元组，二元组内部用'_'连接，二元组之间用'&'分割。"


def generate_prompt(input, instruction):
    if input:
        instruction = instruction + '\n' + input
    return prompt_input.format_map({'instruction': instruction})


if __name__ == '__main__':
    fl_score_list = []
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.lora_model
        if args.lora_model is None:
            args.tokenizer_path = args.base_model
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)

    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=False,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
    )

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
    if model_vocab_size != tokenzier_vocab_size:
        assert tokenzier_vocab_size > model_vocab_size
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenzier_vocab_size)
    if args.lora_model is not None:
        print("loading peft model")
        model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=load_type)
    else:
        model = base_model

    if device == torch.device('cpu'):
        model.float()
    else:
        with open(args.data_file, 'r') as f:
            examples = json.loads(f.read())
        #     examples = [l.strip() for l in f.readlines()]
        # print("first 3 examples:")
        for example in examples[:3]:
            print(example)

    model.to(device)
    model.eval()

    with torch.no_grad():
        if args.interactive:
            while True:
                raw_input_text = input("Input:")
                if len(raw_input_text.strip()) == 0:
                    break
                if args.with_prompt:
                    input_text = generate_prompt(input=raw_input_text,instruction=instruction)
                else:  # todo 测试去掉了
                    input_text = raw_input_text
                inputs = tokenizer(input_text, return_tensors="pt")  # add_special_tokens=False ?
                generation_output = model.generate(
                    input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs['attention_mask'].to(device),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    **generation_config
                )
                s = generation_output[0]
                output = tokenizer.decode(s, skip_special_tokens=True)
                if args.with_prompt:
                    response = output.split("### Response:")[1].strip()
                else:
                    response = output
                print("Response: ", response)
                print("\n")
        else:
            results = []
            for index, example in enumerate(examples):
                if args.with_prompt is True:
                    input_text = example['input']
                    input_text = generate_prompt(input=input_text,instruction=example['instruction'])
                else:
                    input_text = example
                inputs = tokenizer(input_text, return_tensors="pt")  # add_special_tokens=False ?
                generation_output = model.generate(
                    input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs['attention_mask'].to(device),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    **generation_config
                )
                s = generation_output[0]
                output = tokenizer.decode(s, skip_special_tokens=True)
                if args.with_prompt:
                    response = output.split("### Response:")[1].strip().replace("\n", "&")
                else:
                    response = output.replace("\n", "&")
                # 添加评估
                label = 1 if example['output'] == response else 0
                fl_score_list.append(label)
                results.append({"text": example['text'], "ori_answer": example['output'], "gen_answer": response})

            dirname = os.path.dirname(args.predictions_file)
            os.makedirs(dirname, exist_ok=True)
            with open(args.predictions_file, 'w+') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            with open(dirname + '/generation_config.json', 'w+') as f:
                json.dump(generation_config, f, ensure_ascii=False, indent=4)
