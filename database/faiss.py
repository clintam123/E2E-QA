import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from data.load_data import load_data

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Để tạo vector embedding cho câu hỏi, ta sử dụng vector hidden state từ token CLS trong output của model
def cls_pooling(model_output):
    return model_output.hidden_states[-1][:, 0]


def get_embeddings(text_list):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    MODEL_NAME = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input, output_hidden_states=True)
    return cls_pooling(model_output)

def get_embedding_dataset():
    EMBEDDING_COLUMN = "question_embedding"
    raw_datasets = load_data()
    embeddings_dataset = raw_datasets.map(
        lambda x: {
            EMBEDDING_COLUMN: get_embeddings(x["question"]).detach().cpu().numpy()[0]
        }
    )
    embeddings_dataset.add_faiss_index(column=EMBEDDING_COLUMN)
    return embeddings_dataset

