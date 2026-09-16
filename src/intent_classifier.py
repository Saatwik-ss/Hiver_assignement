"""
intent_classifier.py — SetFit-based intent classifier for Uber_Support.

Usage:
  # Train
  python src/intent_classifier.py --train

  # Predict (single)
  python src/intent_classifier.py --predict "my driver cancelled on me"

  # Batch predict (JSON list of texts)
  python src/intent_classifier.py --batch data/test_texts.json
"""

import json
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

MODEL_PATH    = os.getenv("SETFIT_MODEL_PATH", "models/setfit_uber")
FEW_SHOT_PATH = os.getenv("FEW_SHOT_DATA",     "data/few_shot_train.json")
INTENT_MODEL  = os.getenv("INTENT_MODEL",       "sentence-transformers/paraphrase-mpnet-base-v2")


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train(few_shot_path: str = FEW_SHOT_PATH,
          model_save_path: str = MODEL_PATH,
          base_model: str = INTENT_MODEL):
    """Fine-tune SetFit on few-shot labelled examples."""
    from setfit import SetFitModel, Trainer, TrainingArguments
    from datasets import Dataset

    print(f"Loading few-shot data from {few_shot_path} …")
    with open(few_shot_path, encoding="utf-8") as f:
        raw = json.load(f)

    # Keep labels as strings — SetFit handles encoding internally.
    # Only keep text + label columns to avoid confusion.
    clean = [{"text": d["text"], "label": d["label"]} for d in raw]
    dataset = Dataset.from_list(clean)

    labels = sorted(set(dataset["label"]))
    print(f"Classes ({len(labels)}): {labels}")
    print(f"Total training examples: {len(dataset)}")

    # 80/20 split for eval during training
    split     = dataset.train_test_split(test_size=0.2, seed=42)
    train_set = split["train"]
    eval_set  = split["test"]

    # Pass labels explicitly so model.labels uses our canonical sorted order
    model = SetFitModel.from_pretrained(base_model, labels=labels)

    args = TrainingArguments(
        output_dir=model_save_path,
        num_epochs=3,
        num_iterations=40,
        batch_size=32,
        body_learning_rate=1e-5,
        head_learning_rate=1e-2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        metric="accuracy",
    )

    print("Training SetFit (3 epochs × 40 iterations × 12 classes) …")
    trainer.train()

    # Verify model.labels matches our sorted list
    assert list(model.labels) == labels, \
        f"Label mismatch: model={model.labels} vs ours={labels}"

    os.makedirs(model_save_path, exist_ok=True)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {str(i): l for l, i in label2id.items()}
    label_map = {"labels": labels, "label2id": label2id, "id2label": id2label}
    with open(os.path.join(model_save_path, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    model.save_pretrained(model_save_path)
    print(f"Model saved → {model_save_path}")
    print(f"Verified model.labels: {model.labels}")
    return model, labels



# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

class IntentClassifier:
    """Thin wrapper around a trained SetFit model for inference."""

    def __init__(self, model_path: str = MODEL_PATH):
        from setfit import SetFitModel
        self._model = SetFitModel.from_pretrained(model_path)

        label_map_path = os.path.join(model_path, "label_map.json")
        with open(label_map_path) as f:
            maps = json.load(f)

        # Use the 'labels' list (canonical ordering) saved during training.
        # model.labels should match, but we keep our own copy for safety.
        self._labels: list[str] = maps["labels"]
        print(f"IntentClassifier ready — {len(self._labels)} intents: {self._labels}")

    def predict(self, texts: list[str]) -> list[dict]:
        """
        Return list of dicts:
          {intent, confidence, second_intent, second_conf, all_probs}

        Uses model.predict() for labels (ground truth from SetFit) and
        predict_proba() for confidence scores.
        """
        # model.predict() returns the correct string labels (trusted source of truth)
        pred_labels = list(self._model.predict(texts))

        # predict_proba returns a Tensor on whichever device the model is on
        with torch.no_grad():
            raw_probs = self._model.predict_proba(texts).cpu().float()  # (N, C)

        # The probs columns align with model.labels (NOT our self._labels sorted list)
        # Use model.labels for column mapping to be safe
        model_labels = list(self._model.labels)
        probs_np = raw_probs.numpy()  # (N, C)

        results = []
        for i, (pred_label, prob_row) in enumerate(zip(pred_labels, probs_np)):
            sorted_idx  = np.argsort(prob_row)[::-1]
            best_idx    = int(sorted_idx[0])
            second_idx  = int(sorted_idx[1])

            # Confidence = prob at the predicted label's column
            pred_col    = model_labels.index(pred_label) if pred_label in model_labels else best_idx
            confidence  = float(prob_row[pred_col])

            second_int  = model_labels[second_idx]
            second_conf = float(prob_row[second_idx])

            results.append({
                "intent":        pred_label,
                "confidence":    confidence,
                "second_intent": second_int,
                "second_conf":   second_conf,
                "all_probs":     {lbl: float(prob_row[j])
                                  for j, lbl in enumerate(model_labels)},
            })
        return results

    def predict_one(self, text: str) -> dict:
        return self.predict([text])[0]


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Uber intent classifier")
    parser.add_argument("--train",   action="store_true", help="Train SetFit")
    parser.add_argument("--predict", type=str,            help="Predict single text")
    parser.add_argument("--batch",   type=str,            help="Batch predict from JSON file")
    parser.add_argument("--model",   default=MODEL_PATH,  help="Model path")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.predict:
        clf = IntentClassifier(args.model)
        result = clf.predict_one(args.predict)
        print(f"Intent    : {result['intent']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Runner-up : {result['second_intent']} ({result['second_conf']:.1%})")
        print("All probs :")
        for lbl, p in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
            bar = "█" * int(p * 30)
            print(f"  {lbl:<22} {p:5.1%}  {bar}")
    elif args.batch:
        clf = IntentClassifier(args.model)
        with open(args.batch) as f:
            texts = json.load(f)
        results = clf.predict(texts)
        for text, res in zip(texts, results):
            print(f"{res['intent']:25s} ({res['confidence']:.0%})  {text[:80]}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
