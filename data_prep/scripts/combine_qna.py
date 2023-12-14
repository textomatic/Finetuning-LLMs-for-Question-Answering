import os
import json

def flatten_nested_lists(nested_lists):
    """Flattens list of lists containing all elements."""
    flattened_list = []

    # Iterate through the nested lists and add each element to the flattened_list
    for sublist in nested_lists:
        flattened_list.extend(sublist)

    return flattened_list


def main(docs_path, question_json):
    """Combine questions and answers from all documents into a single JSON file for easier processing."""
    all_lists = []
    count = 0

    for file in os.listdir():
        if file.endswith('.result.json'):
            with open(os.path.join(docs_path, file), 'r') as f:
                content = f.read()
                one_list = json.loads(content)
                all_lists.append(one_list)
            count += 1
    print(f"Total {count} .result.json files combined.")
    
    final_list = flatten_nested_lists(all_lists)
    count = len(final_list)

    with open(question_json, 'w') as output_file:
        json.dump(final_list, output_file, indent=4)
        print(f"{count} question-answer pairs have been saved to '{question_json}'.")


if __name__=='__main__':
    main('data_gpt35_v2/docs', 'data_gpt35_v2/questions.json')