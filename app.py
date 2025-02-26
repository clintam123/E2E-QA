from database.faiss import get_embeddings, get_embedding_dataset
from transformers import pipeline

PIPELINE_NAME = 'question-answering'
MODEL_NAME = 'tamdd18/distilbert-finetuned-squadv2'
pipe = pipeline(PIPELINE_NAME, model=MODEL_NAME)

TOP_K = 3
EMBEDDING_COLUMN = "question_embedding"
embeddings_dataset = get_embedding_dataset()
for idx, input_question in enumerate(embeddings_dataset['question'][200:210]):
    input_quest_embedding = get_embeddings([input_question]).cpu().detach().numpy()
    scores, samples = embeddings_dataset.get_nearest_examples(
        EMBEDDING_COLUMN, input_quest_embedding, k=TOP_K
    )
    print(f'Question {idx + 1}: {input_question}')
    for jdx, score in enumerate(scores):
        print(f'Top {jdx + 1}\tScore: {score}')
        context = samples['context'][jdx]
        answer = pipe(
            question=input_question,
            context=context
        )
        print(f'Context: {context}')
        print(f'Answer: {answer}')
        print()
    print()