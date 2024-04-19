from datasets import load_dataset
import json
import os
import tiktoken
from openai import OpenAI
import time
from tqdm import tqdm

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

    user_prompt = prompts["user_prompt"].format(text_to_compress=text)

    res = openai_client.chat.completions.create(
        # model="gpt-4-turbo-2024-04-09",
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompts["system_prompt"]},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        top_p=1.0,
    )

    return res.choices[0].message.content


def create_compression_dataset():
    meeting_data_fp = open("meeting_data.json", "r")
    data = json.load(meeting_data_fp)
    meeting_data_fp.close()

    compression_data = []
    compressed_idxs = set()
    if os.path.exists("compressed_meeting_data.json"):
        ac_meeeting_data_fp = open("compressed_meeting_data.json", "r")
        ac_data = json.load(ac_meeeting_data_fp)
        ac_meeeting_data_fp.close()

        for ac_datum in ac_data:
            compressed_idxs.add(ac_datum["index"])
        compression_data.extend(ac_data)

    last_save_len = 0

    for i, datum in tqdm(enumerate(data), desc="Data Compression ", position=1):
        if i in compressed_idxs:
            continue

        if len(compression_data) % 10 == 0:
            compressed_meeting_data_fp = open("compressed_meeting_data.json", "w")
            json.dump(compression_data, compressed_meeting_data_fp)
            compressed_meeting_data_fp.close()

            last_save_len = len(compression_data)

            print(f"Saved {last_save_len} data points!")

        exit = False

        chunks = get_text_chunks(datum["prompt"])

        while True:
            try:
                print("examples processed this session", len(compression_data))

                new_datum = {"text": datum["prompt"]}
                comps = []
                for chunk in tqdm(chunks, desc="Chunk Compression", position=0):
                    comps.append(compress(chunk))

                new_datum["compressed_text"] = "".join(comps)
                new_datum["text_list"] = chunks
                new_datum["compressed_list"] = comps
                new_datum["index"] = i

                compression_data.append(new_datum)
                break

            except KeyboardInterrupt:
                print("Keyboard interrupt stopped compression.")
                exit = True
                break

            except Exception as err:
                print("Some other error occurred while executing:")
                print(err)

        if exit:
            break

    if len(compression_data) >= last_save_len:
        compressed_meeting_data_fp = open("compressed_meeting_data.json", "r")
        json.dump(compression_data, compressed_meeting_data_fp)
        compressed_meeting_data_fp.close()
        print("Compression Data Saved!")
    else:
        print("Error in pushing compressed data to array.")


def main():
    if not os.path.exists("meeting_data.json"):
        print("Loading Data...")
        load_data()

    print("Data Loaded!")

    print("Compressing\n")
    create_compression_dataset()


if __name__ == "__main__":
    main()
