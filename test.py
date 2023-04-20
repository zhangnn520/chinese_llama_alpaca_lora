import json
from metrics import compute_metrics
ori_list,gen_list = [],[]
with open("/root/autodl-fs/chatglm_finetune/output_dir_pt_20/global_step-300/ft_pt_answer.json", 'r') as f:
    examples = json.loads(f.read())
    for example in examples:
        ori_list.append(example['ori_answer'])
        gen_list.append(example['gen_answer'])
    score_dict = compute_metrics(ori_list,gen_list)
    print(json.dumps(score_dict,ensure_ascii=False,indent=4))