from tqdm.autonotebook import tqdm
import requests
import os

def http_get(url, path):
    """
    Downloads a URL to a given path on disc
    """
    if os.path.dirname(path) != '':
        os.makedirs(os.path.dirname(path), exist_ok=True)

    req = requests.get(url, stream=True)
    if req.status_code != 200:
        print("Exception when trying to download {}. Response {}".format(url, req.status_code), file=sys.stderr)
        req.raise_for_status()
        return

    download_filepath = path+"_part"
    with open(download_filepath, "wb") as file_binary:
        content_length = req.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                progress.update(len(chunk))
                file_binary.write(chunk)

    os.rename(download_filepath, path)
    progress.close()

def convert_to_dataset_torch(data: pandas.DataFrame, labels: pandas.Series) -> TensorDataset:
    input_ids = []
    attention_masks = []
    token_type_ids = []
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        encoded_dict = tokenizer.encode_plus(row["question1"], row["question2"], max_length=max_length, pad_to_max_length=True, 
                      return_attention_mask=True, return_tensors='pt', truncation=True)
        # Add the encoded sentences to the list.
        input_ids.append(encoded_dict['input_ids'])
        token_type_ids.append(encoded_dict["token_type_ids"])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels.values)
    
    return TensorDataset(input_ids, attention_masks, token_type_ids, labels)