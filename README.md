# Fine-tuning Large Language Models for Question-Answering

## Motivation
This project was part of the independent studies course about Large Language Models (LLMs) and Generative AI (GenAI). The aim was to pick one area of LLMs/GenAI to focus on and gain deeper theoretical understanding or application skills. My choice of focus was to learn more about supervised fine-tuning of LLMs and I decided to approach that by working on a problem of manageable magnitude: cars.

Cars are everywhere in this country, but how well do people know their car? If they encounter an issue with their vehicle or have a doubt about some settings in their automobile, do they have to dig out their owners' manual and flip through physical pages of books? Or is there a better way to help car owners in such situations?

I decided to experiment with fine-tuning open-source LLMs such that they are able to answer questions related to a particular make of car. The fine-tuned model would ideally be the virtual assistant that a car owner would like to first approach before attempting anything.

## Data
The car make that I settled on for this project was the Tesla Model 3, which has become increasingly popular on the roads in recent years, thanks to their quality and generous electric vehicle tax credits from the government. 

The [Owner's Manual](https://www.tesla.com/ownersmanual/model3/en_us/) of Tesla Model 3 is available online in web pages or as a Portable Document Format ([PDF](https://www.tesla.com/ownersmanual/model3/en_us/Owners_Manual.pdf)) file.There were thus two ways to obtain data about Model 3 - web scraping and optical character recognition (OCR). I chose the former.

Some manual post-processing of the collected data was done. They include removal of non-textual information, removal of non-ASCII characters, removal of excessive white spaces, and converting the documents into markdown format. There was a total of 128 documents upon completion of post-processing.

## Approach
There are multiple ways to perform supervised fine-tuning of LLMs. I chose to limit the scope of this project by focusing on Parameter Efficient Fine-Tuning (PEFT) methods, particularly the Low-Rank Adapter (LoRA) approach. 

There is a multitude of open-source LLMs nowadays. So, I was spoilt for choice when I had to make a decision on which models to work on. I eventually chose to experiment with Meta's Llama 2 and Mistral AI's Mistral models. For ease of comparison and deployment, the 7-billion-parameter variant of the models were used. 

In order to perform supervised fine-tuning (SFT), a labeled dataset is required. I could either come up with the questions and answers manually or enlist the help of advanced LLMs like GPT-3.5/4 to speed up the process. For the latter, the general idea was to leverage prompt engineering and pass documents to LLMs to extract questions and answers. There are a number of libraries and packages which could help us out on this. The one that I tried out was called [question_extractor](https://github.com/nestordemeure/question_extractor). I had to make some modifications to their scripts in order to work with Azure OpenAI endpoints as their scripts were intended for use with OpenAI endpoints only. A total of 2078 question-answer pairs were extracted.

Upon getting the question-answer data, the next step was to ensure they are in the appropriate format to be used for SFT. Some additional work needed to be done to fit every question-answer pair into a format that looked like this:
```
<s>[INST] question [/INST] answer </s>
```

The Llama 2 and Mistral models contained more than 7-billion parameters and were gigantic in size. In order to perform SFT on them realistically, I had to resort to using techniques like LoRA or QLoRA (which refers to quantized LoRA). As HuggingFace has the weights of these open-source models, I decided to leverage their ecosystem of libraries which included `transformers`, `accelerate`, `peft`, `trl`, and `bitsandbytes`. The `SFTTrainer` was the main class used for performing supervised fine-tuning.

There are a lot of parameter values that needed to be decided on when using LoRA/QLoRA. Some of the key ones are rank, alpha scaling parameter, dropout probability, quantization type, etc. Trial and error was the approach taken in this stage to learn and find out what worked and what did not, especially when a specific dataset was involved. For details about the parameter values used in training the models, please refer to the config json files in the repository.

## Deployment
Both models were trained for a total of 10 epochs each. And upon completion of training, the adapter weights were merged with the weight of their base models. The merged models were then pushed to HuggingFace hub for ease of subsequent deployment in various public cloud platforms. 

The two fine-tuned models are:
- [llama-2-7b-tsla-qna-4](https://huggingface.co/textomatic/llama-2-7b-tsla-qna-4)
- [mistral-7b-tsla-qna-1](https://huggingface.co/textomatic/mistral-7b-tsla-qna-1)

Some other models were trained along the way as well, but the most promising candidates are the above two. 

The models were subsequently deployed on AWS Sagemaker as inference endpoints. Virtual machine instances were stood up in Azure to host Streamlit web applications that acted as the front end for interacting with the fine-tuned models. In the backend, the instances were communicating with the AWS endpoints to obtain answers to user questions. 

To provide a point of comparison, a Retrieval Augmented Generation (RAG) solution was developed and deployed in Azure. [Haystack](https://github.com/deepset-ai/haystack) was used for orchestrating the RAG pipeline while Streamlit served as the frontend application. The RAG solution finds the most promising candidate text(s) from the document store and pass them together with the user's query to LLMs to obtain an answer. The LLMs used in this RAG pipeline are the pre-trained variants of Llama 2 7B and Mistral 7B. 

## Results
As there does not yet exist any industrial standard for systematic evaluation of text generation models, only qualitative testing and manual evaluation were performed in this project. The results were promising as both Llama 2 7B and Mistral 7B had less than 1% of their total parameters trained but they were able to retain new knowledge. Check out the demo video to see the fine-tuned models in action!

## Future work
There are certainly many areas of improvement that this project could take on moving forward:
- Fine-tuning models that are smaller than 7-billion parameters
- Quantization and deployment on edge devices
- Fine-tuning of multi-modal models such that questions and answers could be conveyed not only through words but also visuals (this would be particularly helpful as a lot of tips and instructions shared in the owners' manual were in the form of a picture or animation)

## Sitemap

```
.
|____fine-tune
| |____deploy
| | |____requirements.txt
| | |____.streamlit
| | | |____config.toml
| | |____.gitignore
| | |____app.py
| | |____assets
| | | |____icon-tesla-48x48.png
| |____scripts
| | |____config_mistral_7b.json
| | |____requirements.txt
| | |____config_llama_7b.json
| | |____fine-tune_llm.py
|____data_prep
| |____scripts
| | |____combine_qna.py
| | |____generate_jsonl.py
| |____data
| | |____question_answer_pairs.jsonl
| | |____docs
| | | |____identification_labels.md
| | | |____cold_weather_best_practices.md
.......
| | | |____troubleshooting_alerts_p1.md
| | | |____track_mode.md
| | |____questions.json
|____rag
| |____deploy
| | |____requirements.txt
| | |____.streamlit
| | | |____config.toml
| | |____.gitignore
| | |____utils.py
| | |____app.py
| | |____data
| | | |____identification_labels.md
| | | |____cold_weather_best_practices.md
| | | |____installing_phone_charging_cable.md
.......
| | | |____troubleshooting_alerts_p1.md
| | | |____track_mode.md
| | |____assets
| | | |____icon-tesla-48x48.png
|____README.md
|____.gitignore
```