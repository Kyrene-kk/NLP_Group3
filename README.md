# Group3_Machine_Generated_Text_Detection
## File Description

```
baseline: source code for fine-tuning models like baseline model xlm-roberta-base, LLM models Mistral-7B etc. 

prediction: the predicition results in jsonl format of the fine-tuned models and metric-based methods.

Group3_Machine_Generated_Text_Detection.ipynb: Ensemble the results from different model for voting system.
```
## Code usage
1. For the metric-based method, we use from [MGTBench](https://github.com/xinleihe/MGTBench) and [IMGTB](https://github.com/kinit-sk/IMGTB).
2. For model fine-tuning, run the following code for baseline and LLMs such as Mistral, respectively.
```
# fine-tune xlmRoberta-base
python3 baseline/transformer_baseline.py --train_file_path data/subtaskA_train_multilingual.jsonl --test_file_path data/subtaskA_test_multilingual.jsonl --prediction_file_path predictions/roberta_test_predictions_probs.jsonl --subtask A --model 'xlm-roberta-base'

#fine-tune Mistral-7B, Llama-7B or Falcon-7B
python3 baseline/transformer_peft.py --train_file_path data/subtaskA_train_multilingual.jsonl --test_file_path data/subtaskA_test_multilingual.jsonl --prediction_file_path predictions/mistral_test_predictions_probs.jsonl --subtask A --model 'mistralai/Mistral-7B-v0.1'
```
3. For system-based method, use the provided jupyter notebook with the files from the prediction directory to generate the quantitative results.
## Parameters setting for fine-tuning with lora
```
lora_alpha = 16
lora_dropout = 0.1
lora_r = 2
num_train_epochs=3
```
