from datasets import load_dataset
import json
import os
import tiktoken
from openai import OpenAI
import time

tokenizer = tiktoken.encoding_for_model("gpt-4")
openai_client = OpenAI()
prompts = json.load(open("prompts.json", "r"))


def load_data():
    dataset = load_dataset("huuuyeah/meetingbank", split="train")
    data = []
    for idx, instance in enumerate(dataset):
        data.append(
            {
                "index": idx,
                "prompt": instance["transcript"],
                "summary": instance["summary"],
            }
        )

    json.dump(data, open("meeting_data.json", "w"), indent=2)


def get_text_chunks(full_text, chunk_size=512):
    chunks = []
    tok_full_text = tokenizer.encode(full_text)
    end_toks = set(tokenizer.encode(".") + tokenizer.encode("\n"))
    text_len = len(tok_full_text)
    i = 0

    while i < text_len:
        if (
            text_len - i <= chunk_size
        ):  # if length of remaining text is <= chunksize finish loop
            chunks.append(tokenizer.decode(tok_full_text[i:]))
            break
        else:
            j = i + chunk_size
            for k in range(i + chunk_size - 1, i - 1, -1):
                if tok_full_text[k] in end_toks:
                    j = k
                    break
            chunks.append(tokenizer.decode(tok_full_text[i : j + 1]))
            i = j + 1

    return chunks


def compress(text):

    prompts["user_prompt"] = prompts["user_prompt"].format(text_to_compress=text)

    res = openai_client.chat.completions.create(
        # model="gpt-4-turbo-2024-04-09",
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompts["system_prompt"]},
            {"role": "user", "content": prompts["user_prompt"]},
        ],
        temperature=0.3,
        top_p=1.0,
    )

    return res.choices[0].message.content


def create_compression_dataset():
    data = json.load(open("meeting_data.json", "r"))

    compression_data = []

    start_time = time.time()
    for i, datum in enumerate(data):
        prev_time = time.time()
        print(
            f"datapoint {i}",
            f"{i / len(data)}% Compression Completed",
            f"compressed in {prev_time - start_time} seconds",
        )

        if i == 2:
            break

        new_datum = {"text": datum["prompt"]}
        chunks = get_text_chunks(datum["prompt"])
        comps = []
        for j, chunk in enumerate(chunks):
            print(f"{j/len(chunks)} completed")
            try:
                comps.append(compress(chunk))
            except Exception as e:
                print(e)
                break

        new_datum["compressed_text"] = "".join(comps)
        new_datum["text_list"] = chunks
        new_datum["compressed_list"] = comps

    print("100% Compression Completed")
    print("Saving Compression Data...")

    compression_data.append(new_datum)

    json.dump(compression_data, open("compressed_meeting_data.json", "w"), indent=2)

    print("Compression Data Saved!")


def main():
    if not os.path.exists("meeting_data.json"):
        print("Loading Data...")
        load_data()

    print("Data Loaded!")

    print("Compressing")
    create_compression_dataset()


if __name__ == "__main__":
    main()
