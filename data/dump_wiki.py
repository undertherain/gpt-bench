import json

from datasets import load_dataset

corpus = load_dataset("iohadrubin/wikitext-103-raw-v1", split="train", streaming=True)

with open(f"corpus.jsonl", "w") as file_out:
    # for doc in tqdm(documents[:10]):
    cnt = 0
    for entry in iter(corpus):
        text = entry["text"]
        text = text.replace("<br />", "\n")
        text = text.replace("\n\n", "\n")
        text = text.replace('\\', "")
        if len(text) > 600:
            cnt += 1
            data = {"text": text}
            file_out.write(json.dumps(data))
            file_out.write("\n")
        if cnt > 10000:
            break
