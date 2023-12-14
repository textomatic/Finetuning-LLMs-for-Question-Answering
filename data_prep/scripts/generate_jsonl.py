import json

def main(questions_json, qa_pairs_jsonl):
    """Function to format data for fine-tuning"""
    count = 0
    jsonl_str = ''

    with open(questions_json, 'r') as f:
        content = f.read()
        question_list = json.loads(content)
    
    for qa_pair in question_list:
        qa_dict = {"text": "<s>[INST] " + qa_pair["question"] + " [/INST] " + qa_pair["answer"] + "</s>"}
        jsonl_str += json.dumps(qa_dict) + "\n"
        count += 1
    
    with open(qa_pairs_jsonl, 'w') as f:
        f.write(jsonl_str)
        print(f"Total {count} question-answer pairs written to jsonl file.")


if __name__=='__main__':
    main('data_gpt35_v2/questions.json', 'data_gpt35_v2/question_answer_pairs.jsonl')