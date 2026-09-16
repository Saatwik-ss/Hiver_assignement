---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: accidentally left medicine in the uber, I need it urgently
- text: why was I charged a cancellation fee when the driver cancelled
- text: driver says he doesn't have my item but I definitely left it there
- text: driver cancelled three times in a row
- text: forgot my password and can't reset it
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
base_model: sentence-transformers/paraphrase-mpnet-base-v2
---

# SetFit with sentence-transformers/paraphrase-mpnet-base-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [sentence-transformers/paraphrase-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/paraphrase-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 512 tokens
- **Number of Classes:** 12 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label             | Examples                                                                                                                                                                                                       |
|:------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| trip_issue        | <ul><li>'ended up at completely wrong address'</li><li>'driver kept missing turns and the route was terrible'</li><li>'the GPS navigation was wrong and driver followed it anyway'</li></ul>                   |
| driver_behavior   | <ul><li>'driver made me feel uncomfortable with his behaviour'</li><li>'driver was extremely rude and unprofessional'</li><li>'driver kept making inappropriate comments throughout the ride'</li></ul>        |
| account_access    | <ul><li>"I can't verify my identity to complete account setup"</li><li>"I can't update my phone number on my account"</li><li>'I think someone else is using my uber account'</li></ul>                        |
| safety_incident   | <ul><li>'driver threatened me when I gave a low rating'</li><li>'driver locked the doors and refused to let me out'</li><li>'my driver sexually harassed me during the ride'</li></ul>                         |
| ride_cancellation | <ul><li>'driver accepted then immediately cancelled the ride'</li><li>'multiple driver cancellations in the same evening'</li><li>'I got charged a cancellation fee but the driver cancelled not me'</li></ul> |
| eats_order        | <ul><li>'my food was damaged when the delivery arrived'</li><li>'uber eats delivered food from the wrong restaurant'</li><li>'delivery driver left my food at the wrong address'</li></ul>                     |
| payment_billing   | <ul><li>'receipt shows wrong fare amount compared to what I agreed to'</li><li>'requesting a price adjustment for my recent trip'</li><li>"I need a refund for a ride I didn't take"</li></ul>                 |
| app_technical     | <ul><li>"I can't request a ride, the button is greyed out"</li><li>"app won't load and shows a blank screen"</li><li>'the uber app is not showing any available drivers'</li></ul>                             |
| general_complaint | <ul><li>'your service has been terrible lately'</li><li>"I'm extremely unhappy with my uber experience"</li><li>'uber has really gone downhill, very frustrated'</li></ul>                                     |
| wait_time         | <ul><li>'driver is showing as arrived but is not here'</li><li>'the estimated arrival time is completely wrong'</li><li>'my driver is 20 minutes away but the app showed 5 minutes'</li></ul>                  |
| positive_feedback | <ul><li>'driver was so polite and the car was spotless'</li><li>'uber came in exactly on time and the ride was perfect'</li><li>'the new uber features are great, much better experience'</li></ul>            |
| lost_item         | <ul><li>'forgot my headphones in the uber earlier today'</li><li>'I lost my wallet in the uber, can you help me retrieve it'</li><li>'lost something in my last uber trip, please help'</li></ul>              |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("driver cancelled three times in a row")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median | Max |
|:-------------|:----|:-------|:----|
| Word count   | 5   | 9.3007 | 14  |

| Label             | Training Sample Count |
|:------------------|:----------------------|
| account_access    | 13                    |
| app_technical     | 12                    |
| driver_behavior   | 13                    |
| eats_order        | 10                    |
| general_complaint | 14                    |
| lost_item         | 13                    |
| payment_billing   | 12                    |
| positive_feedback | 12                    |
| ride_cancellation | 12                    |
| safety_incident   | 15                    |
| trip_issue        | 13                    |
| wait_time         | 14                    |

### Training Hyperparameters
- batch_size: (32, 32)
- num_epochs: (3, 3)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 40
- body_learning_rate: (1e-05, 1e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- evaluation_strategy: epoch
- eval_max_steps: -1
- load_best_model_at_end: True

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0026 | 1    | 0.1501        | -               |
| 0.1305 | 50   | 0.1652        | -               |
| 0.2611 | 100  | 0.1155        | -               |
| 0.3916 | 150  | 0.0585        | -               |
| 0.5222 | 200  | 0.0296        | -               |
| 0.6527 | 250  | 0.0168        | -               |
| 0.7833 | 300  | 0.0078        | -               |
| 0.9138 | 350  | 0.0047        | -               |
| 1.0    | 383  | -             | 0.0591          |
| 1.0444 | 400  | 0.0028        | -               |
| 1.1749 | 450  | 0.0020        | -               |
| 1.3055 | 500  | 0.0015        | -               |
| 1.4360 | 550  | 0.0013        | -               |
| 1.5666 | 600  | 0.0011        | -               |
| 1.6971 | 650  | 0.0011        | -               |
| 1.8277 | 700  | 0.0010        | -               |
| 1.9582 | 750  | 0.0009        | -               |
| 2.0    | 766  | -             | 0.0579          |
| 2.0888 | 800  | 0.0009        | -               |
| 2.2193 | 850  | 0.0008        | -               |
| 2.3499 | 900  | 0.0008        | -               |
| 2.4804 | 950  | 0.0008        | -               |
| 2.6110 | 1000 | 0.0007        | -               |
| 2.7415 | 1050 | 0.0007        | -               |
| 2.8721 | 1100 | 0.0008        | -               |
| 3.0    | 1149 | -             | 0.0568          |

### Framework Versions
- Python: 3.11.5
- SetFit: 1.2.0
- Sentence Transformers: 5.3.0
- Transformers: 5.11.0
- PyTorch: 2.5.1+cu121
- Datasets: 4.3.0
- Tokenizers: 0.22.2

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->