"""
for inference
"""
from typing import Any, Dict
from transformers import AutoTokenizer
import torch
import numpy as np
import json
import random
import re

import more_itertools
from src.utils.dataset_utils import pad2sameLen
from DPR.dpr.utils.tasks import task_map, get_prompt_files
from DPR.dpr.utils.data_utils import read_data_from_json_files

from itertools import permutations
def tune_ordering(p_list, id): #id in [0,5]
    all_permutations = list(permutations(p_list))
    return all_permutations[id]

def remove_double_space(string):
    return re.sub("[ ]{2,}", " ", string)


class FewShotDatasetReader(torch.utils.data.Dataset):
    def __init__(
        self,
        model_name,
        task_name,
        prompt_file,
        prompt_pool_path,
        shot_num=-1,
        n_tokens=1600,
        random_sample=False,
        random_seed=0,
        cache_dir=None,
        max_length=2048,
        train_clusters=None,
        order_id=0,
    ) -> None:
        self.task = task_map.cls_dic[task_name]()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, model_max_length=max_length,
        )
        if self.task.class_num == 1:
            self.tokenizer.padding_side = "left"

        # retreived prompt_file,
        with open(prompt_file) as f:
            self.prompts = json.load(f)

        # random sample from prompt pool
        self.random_sample = random_sample
        if random_sample:
            prompt_pool_path = get_prompt_files(prompt_pool_path, train_clusters)
            print("prompt files: ", prompt_pool_path)
            self.prompt_pool = read_data_from_json_files(prompt_pool_path)
            print("prompt passages num : ", len(self.prompt_pool))
            self.random_seed = random_seed

        self.shot_num = shot_num
        self.n_tokens_in_prompt = n_tokens
        self.num_processes = 1
        self.process_index = 0
        self.order_id = order_id
        
    def __getitem__(self, index):
        if self.task.class_num == 1:
            return self.text_to_instance_completion(self.prompts[index])
        else:
            return self.text_to_instance_choice(self.prompts[index])

    def __len__(self):
        return len(self.prompts)

    def shard(self, accelerator):
        self.num_processes = accelerator.num_processes
        self.process_index = accelerator.process_index
        self.prompts = list(
            more_itertools.distribute(accelerator.num_processes, self.prompts)[
                accelerator.process_index
            ]
        )

    def get_length(self, text):
        tokenized_example = self.tokenizer.encode_plus(
            text, truncation=False, return_tensors="pt"
        )
        shape = tokenized_example.input_ids.squeeze().shape
        if len(shape) == 0:
            return 1
        else:
            return int(shape[0])

    def format_prompt(self, entry):
        prompt_task = task_map.cls_dic[entry["task_name"]]()
        prompt_question = prompt_task.get_question(entry)
        prompt_answer = prompt_task.get_answer(entry)
        qa=f'{prompt_question}{prompt_answer}'
        return remove_double_space(qa)

    def get_fields(self, entry):
        example = entry["meta_data"]
        questions = [entry["instruction"]] * self.task.class_num # for classification task, the instruction have included examples
        answers = self.task.get_answers(example)
        label = self.task.get_label(example)
        if self.shot_num == 0: # TODO: Optimize code logic here
            prompts_list = questions[0].split('\n')
            if len(prompts_list) <=1: prompts_list = []
            else: prompts_list = prompts_list[:-1]
        elif self.random_sample:
            random.seed(self.random_seed)
            prompts_list = [
                p for p in random.choices(self.prompt_pool, k=self.shot_num)
            ]
        else:
            prompts_list = [p["meta_data"] for p in entry["ctxs"]]
        
        if(self.order_id!=0):
            prompts_list = tune_ordering(prompts_list, self.order_id)

        if self.shot_num == 0 and len(prompts_list) > 0: # se2 prompts
            lengths_list = [self.get_length(prompt) for prompt in prompts_list][::-1]
            max_q_length = 0
            max_a_length = 0
            for i in range(len(questions)):
                max_q_length = max(
                    max_q_length, self.get_length(remove_double_space(questions[i].split('\n')[-1:][0]))
                )
                max_a_length = max(
                    max_a_length, self.get_length(remove_double_space(answers[i]))
                )

            max_prompts = np.searchsorted(
                np.cumsum(lengths_list),
                self.n_tokens_in_prompt - len(prompts_list) - (max_q_length + max_a_length),
            )
            prompts_list = prompts_list[-max_prompts:]
            prompt_enc_text = " \n ".join(
                [prompt for prompt in prompts_list]
            )
            questions = [
                remove_double_space(prompt_enc_text + " \n " + question.split('\n')[-1:][0])
                for question in questions
            ]
            return questions, answers, label, len(prompts_list)
        # other retriever prompts 
        lengths_list = [self.get_length(self.format_prompt(prompt)) for prompt in prompts_list]

        max_q_length = 0
        max_a_length = 0
        for i in range(len(questions)):
            max_q_length = max(
                max_q_length, self.get_length(remove_double_space(questions[i]))
            )
            max_a_length = max(
                max_a_length, self.get_length(remove_double_space(answers[i]))
            )

        max_prompts = np.searchsorted(
            np.cumsum(lengths_list),
            self.n_tokens_in_prompt - (max_q_length + max_a_length),
        )
        if self.shot_num > -1:
            max_prompts = min(
                self.shot_num, max_prompts
            )
        
        trunc_prompts_list = prompts_list[:max_prompts][::-1]

        
        prompt_enc_text = " \n ".join(
            [self.format_prompt(prompt) for prompt in trunc_prompts_list]
        )
    
        if max_prompts == 0:
            questions = [remove_double_space(question) for question in questions]
        else:
            questions = [
                remove_double_space(prompt_enc_text + " \n " + question)
                for question in questions
            ]
        return questions, answers, label, max_prompts

    def text_to_instance_choice(self, entry: Dict[str, Any]):
        """
        multiple-choice question
        """
        questions, answers, label, max_prompts = self.get_fields(entry)
        input_ids_list = []
        input_atten_mask_list = []
        input_loss_mask_list = []

        example = {}
        example["enc_text"] = []
        example["enc_answer"] = []
        for i in range(len(questions)):
            enc_text = remove_double_space(questions[i] + answers[i])
            example["enc_text"].append(
                remove_double_space(questions[i]).strip()
            )  # remove trailing space after question
            tokenized_example = self.tokenizer.encode_plus(
                enc_text,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=False,
            )
            enc_answer = remove_double_space(remove_double_space(answers[i]))
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

            example["enc_answer"].append(enc_answer)
        example["n_prompts"] = str(max_prompts)
        example["label"] = label

        return {
            "input_ids": pad2sameLen(input_ids_list, pad_idx=self.tokenizer.pad_token_id),
            "input_atten_mask": pad2sameLen(input_atten_mask_list, pad_idx=0),
            "input_loss_mask": pad2sameLen(input_loss_mask_list, pad_idx=0),
            "labels": torch.tensor([label]),
            "metadata": example,
        }

    def text_to_instance_completion(self, entry: Dict[str, Any]):
        """
        text completion question
        """
        questions, answers, label, max_prompts = self.get_fields(entry)

        input_ids_list = []
        input_atten_mask_list = []
        example = {}
        example["enc_text"] = []
        example["enc_answer"] = []
        for i in range(len(questions)):
            enc_text = remove_double_space(questions[i])
            example["enc_text"].append(
                remove_double_space(questions[i]).strip()
            )  # remove trailing space after question

            tokenized_example = self.tokenizer.encode_plus(
                enc_text,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=False,
            )
            enc_answer = remove_double_space(remove_double_space(answers[i]))

            input_ids = tokenized_example.input_ids.squeeze()
            input_atten_mask = tokenized_example.attention_mask.squeeze()

            if len(input_ids.shape) == 0:
                input_ids = input_ids.unsqueeze(0)
                input_atten_mask = input_atten_mask.unsqueeze(0)

            input_ids_list.append(input_ids)
            input_atten_mask_list.append(input_atten_mask)

            example["enc_answer"].append(enc_answer)
        example["temp_label"] = label
        example["n_prompts"] = str(max_prompts)
        example["label"] = label
        return {
            "input_ids": pad2sameLen(
                input_ids_list, pad_idx=self.tokenizer.pad_token_id, left_pad=True
            ),
            "input_atten_mask": pad2sameLen(
                input_atten_mask_list, pad_idx=0, left_pad=True
            ),
            "metadata": example,
        }

    def gen_to_class_process(self, entry: Dict[str, Any]):
        """
        generate 1 token to classification
        """
        questions, answers, label, max_prompts = self.get_fields(entry)

        input_ids_list = []
        input_atten_mask_list = []
        label_ids_list = []
        example = {}
        example["enc_text"] = []
        example["enc_answer"] = []
        for i in range(len(questions)):
            if i == 0:
                enc_text = remove_double_space(questions[i])
                example["enc_text"].append(
                    remove_double_space(questions[i]).strip()
                )  # remove trailing space after question

                tokenized_example = self.tokenizer.encode_plus(
                    enc_text,
                    truncation=False,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                input_ids = tokenized_example.input_ids.squeeze()
                input_atten_mask = tokenized_example.attention_mask.squeeze()

                if len(input_ids.shape) == 0:
                    input_ids = input_ids.unsqueeze(0)
                    input_atten_mask = input_atten_mask.unsqueeze(0)

                input_ids_list.append(input_ids)
                input_atten_mask_list.append(input_atten_mask)

            enc_answer = remove_double_space(remove_double_space(answers[i]))
            label_encoded = self.tokenizer.encode_plus(
                enc_answer,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"]
            label_id = label_encoded[0]
            label_str = self.tokenizer.convert_ids_to_tokens(label_id)
            if len(label_encoded) > 1:
                logger.warning(
                    f"Cannot find matching id for {enc_answer}, using prefix {label_str}"
                )
            label_ids_list.append(label_id)
        label_ids = torch.tensor(label_ids_list, dtype=torch.long)
        example["n_prompts"] = str(max_prompts)
        example["label"] = label
        return {
            "input_ids": pad2sameLen(input_ids_list, pad_idx=self.tokenizer.pad_token_id, left_pad=True),
            "input_atten_mask": pad2sameLen(input_atten_mask_list, pad_idx=0, left_pad=True),
            "label_ids": label_ids,
            "labels": torch.tensor([label]),
            "metadata": example,
        }
