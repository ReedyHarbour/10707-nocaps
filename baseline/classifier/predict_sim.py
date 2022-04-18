import torch
from transformers import BertTokenizer, VisualBertModel


def predict(text):
    infer_dataloader

    model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    for batch in tqdm(infer_dataloader):
        batch = {key: value.to(device) for key, value in batch.items()}

        with torch.no_grad():
            # shape: (batch_size, max_caption_length)
            # Pass finite state machine and number of constraints if using CBS.
            batch_predictions = model(**batch)["logits"].argmax(-1)
                

        for i, image_id in enumerate(batch["image_id"]):
            instance_predictions = batch_predictions[i, :]

            # De-tokenize caption tokens and trim until first "@@BOUNDARY@@".
            caption = [vocabulary.get_token_from_index(p.item()) for p in instance_predictions]
            eos_occurences = [j for j in range(len(caption)) if caption[j] == "@@BOUNDARY@@"]
            caption = caption[: eos_occurences[0]] if len(eos_occurences) > 0 else caption

            predictions.append({"image_id": image_id.item(), "caption": " ".join(caption)})
