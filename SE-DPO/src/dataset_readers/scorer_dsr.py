from typing import Any, Dict
from transformers import AutoTokenizer
import torch
import pandas as pd
import json
import pandas as pd
from datasets import Dataset
import json
import re
from src.utils.dataset_utils import pad2sameLen_test
from DPR.dpr.utils.tasks import task_map
from tqdm import tqdm
def remove_double_space(string):
    return re.sub("[ ]{2,}", " ", string)

class ScorerDatasetReader(torch.utils.data.Dataset):
    def __init__(
        self,
        example_file,
        model_name,
        task_name,
        prompt_pool_path=None,
        cache_dir=None,
        max_length=2048,
        few_shot_num=3,
    ) -> None:
        self.task = task_map.cls_dic[task_name]()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, model_max_length=self.max_length
        )
        if self.task.class_num == 1:  # text completion question
            self.tokenizer.padding_side = "left"

        # prompt_pool
        with open(prompt_pool_path, "r", encoding="utf-8") as f:
            prompt_pool = json.load(f)
        self.prompt_pool = list(enumerate(prompt_pool)) # tuple (id, dict)
        
        # task_data
        with open(example_file) as f1:
            self.task_data = json.load(f1)

        self.few_shot_num = few_shot_num

        def get_instance_old(entry):
            examples = entry.pop("step_1_ctxs")
            for exp in examples:
                exp.update(self.prompt_pool[exp["id"]][1])
                for key, val in entry.items():
                    exp[f"test_{key}"] = val
            yield from examples

        def get_instance(entry):
            data = []
            for i in range(2, self.few_shot_num+1): # 3-shot now
                if "step_"+str(i)+"_have_choosen" not in entry: continue
                choosen_list = entry["step_"+str(i)+"_have_choosen"]
                sample_list = entry["step_"+str(i)+"_ctxs"]
                for j in range(len(choosen_list)):
                    for id in sample_list[j]:
                        instance = {}
                        instance["have_choosen"] = choosen_list[j]
                        instance.update(self.prompt_pool[id][1])
                        for key, val in entry.items():
                            if(key[:4]=="step"): continue
                            instance[f"test_{key}"] = val
                        data.append(instance)
            yield from data

        def get_dataset(data):
            for entry in data:
                if "step_1_have_choosen" in entry: yield from get_instance_old(entry)
                else: yield from get_instance(entry)

        df = pd.DataFrame(list(get_dataset(self.task_data)))
        self.dataset = Dataset.from_pandas(df)

    def shard(self, accelerator):
        self.dataset = self.dataset.shard(
            num_shards=accelerator.num_processes, index=accelerator.process_index
        )
        self.filter_valid_length()

    def filter_valid_length(self):
        selected_ids = []
        init_len = len(self.dataset)
        for i in tqdm(range(init_len),desc=f"filter by max length of {self.max_length}"):
            entry = self[i]
            if not entry["input_ids"].shape[-1]>=self.max_length:
                selected_ids.append(i)
        self.dataset = self.dataset.select(selected_ids[::-1])
        print(f"{len(self.dataset)}/{init_len}")

    def __getitem__(self, index):
        if self.task.class_num == 1: # text completion question
            return self.text_to_instance_completion(self.dataset[index])
        else:
            return self.text_to_instance_choice(self.dataset[index])

    def __len__(self):
        return len(self.dataset)

    def get_fields(self, entry):
        demonstration = "" 
        if "have_choosen" in entry:
            example_id_list = entry["have_choosen"]
            for example_id in example_id_list:
                example = self.prompt_pool[example_id][1]
                question = self.task.get_question(example)
                answer = self.task.get_answer(example)
                demonstration += remove_double_space(f'{question}{answer}') + " \n "

        # example to be scored 
        question = self.task.get_question(entry)
        answer = self.task.get_answer(entry)
        demonstration = remove_double_space(f'{question}{answer}') + " \n " + demonstration

        test_info = {}
        for key, val in entry.items():
            if key.startswith("test_"):
                test_info[key[len("test_") :]] = val
        test_input_strs = self.task.get_input_strs(test_info)
        test_questions = [demonstration + input for input in test_input_strs]
        test_answer_strs = self.task.get_answers(test_info)
        test_label = self.task.get_label(test_info)
        return test_questions, test_answer_strs, test_label


    def text_to_instance_choice(self, entry):
        """
        multiple-choice question
        """
        test_questions, test_answers, test_label = self.get_fields(entry)  

        input_ids_list = []
        input_atten_mask_list = []
        input_loss_mask_list = []
        for i in range(len(test_questions)):
            enc_text = remove_double_space(test_questions[i] + test_answers[i])
            enc_answer = remove_double_space(test_answers[i])
            tokenized_example = self.tokenizer.encode_plus(
                enc_text,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )
            tokenized_answer = self.tokenizer.encode_plus(
                enc_answer,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )


            answer_mask = tokenized_answer.attention_mask.squeeze()
            if len(answer_mask.shape) == 0:
                answer_mask = torch.tensor([1]).to(answer_mask)

            input_ids = tokenized_example.input_ids.squeeze()
            input_atten_mask = tokenized_example.attention_mask.squeeze()
            input_loss_mask = torch.nn.functional.pad(
                answer_mask, (input_ids.shape[-1] - answer_mask.shape[-1], 0)
            )

            input_ids_list.append(input_ids)
            input_atten_mask_list.append(input_atten_mask)
            input_loss_mask_list.append(input_loss_mask)

        return {
            "input_ids": pad2sameLen_test(input_ids_list, pad_idx=self.tokenizer.pad_token_id, max_length=self.max_length),
            "input_atten_mask": pad2sameLen_test(input_atten_mask_list, pad_idx=0, max_length=self.max_length),
            "input_loss_mask": pad2sameLen_test(input_loss_mask_list, pad_idx=0, max_length=self.max_length),
            "labels": torch.tensor([test_label]),
            "metadata": entry,
        }

    def text_to_instance_completion(self, entry: Dict[str, Any]):
        """
        text completion question
        """
        test_questions, _, test_label = self.get_fields(entry)

        input_ids_list = []
        input_atten_mask_list = []
        for i in range(len(test_questions)): # len(test_questions) = 1 for completion question
            enc_text = remove_double_space(test_questions[i]).strip() 
            tokenized_example = self.tokenizer.encode_plus(
                enc_text,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=False,
            )

            input_ids = tokenized_example.input_ids.squeeze()
            input_atten_mask = tokenized_example.attention_mask.squeeze()

            input_ids_list.append(input_ids)
            input_atten_mask_list.append(input_atten_mask)

        entry["temp_label"] = test_label  # pass label for the next step
        return {
            "input_ids": pad2sameLen_test(
                input_ids_list, pad_idx=self.tokenizer.pad_token_id, max_length=self.max_length, left_pad=True
            ),
            "input_atten_mask": pad2sameLen_test(
                input_atten_mask_list, pad_idx=0, max_length=self.max_length, left_pad=True
            ),
            "metadata": entry,
        }
