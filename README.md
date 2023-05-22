# SecondYearProject
The following project is created in connection with our Second Year Project "Optimizing for cross-lingual learning for multilingual language models on unseen languages of similar structures".
The goal of our project is to answer the following research question: __How can large multilingual language models be optimized for transf learning from seen to unseen languages?__, as well as the following sub-questions:
- SQ1: Does discriminate learning rates for different layers improve cross-lingual performance?
- SQ2: How does similarity of languages impact the transfer of learning?

## Recreating model training
To recreate the model training, follow these steps:
1. Clone the repository
1. Create a virtual environment and install the requirements using eg. using pip:
    ```bash
    pip install -r requirements.txt
    ```
1. To train the baseline model run:
    ```bash
    python3 train.py
    ```
1. To train the other variations use the following arguments to augment the training script:
    - To add discriminate learning rates for different layers:
        ```bash
        --discriminative_lr True
        ```
    - Cosine scheduling:
        ```bash
        --cosine_schedule True
        ```
    Example:
    ```bash
    python3 train.py --discriminative_lr True --cosine_schedule True
    ```

Other hyperparameters can be used by using the following arguments:
- batch_size: `--batch_size`
- learning_rate: `--lr`
- epochs: `--epochs`


## Recreating significance testing of models
1. To replicate the evaluation results used for significance testing of the model optimizations run
    ```bash
    python3 full_run_script.py --replicate True
    ```
    Else if wanting to run experiment with different seeds then run
    ```bash
    python3 full_run_script.py --replicate False
    ```
    When running with different seeds the number of seeds to be ran can be specified using the argument `--num_seeds`. The default is 10.

    All results will be saved to the results folder
1. To get the results for the significance testing run
    ```bash
    python3 significance_testing.py
    ```
    This will print the results of the significance testing to the terminal as well as save the results to a latex table in the folder results

## Recreating model evaluation
To recreate the model evaluation, follow these steps:
1. To evaluate the baseline model run:
    ```bash
    python3 eval.py
    ```
1. To evaluate the other variations use the following arguments to augment the evaluation script:
    - To evaluate the model with discriminate learning rates for different layers:
        ```bash
        --discriminative_lr True
        ```
    - Cosine scheduling:
        ```bash
        --cosine_schedule True
        ```
    Example:
    ```bash
    python3 train.py --discriminative_lr True --cosine_schedule True
    ```
Other hyperparameters can be used by using the following arguments:
- batch_size: `--batch_size`
- save_name: `--save_name` (This will override `--discriminative_lr` and `--cosine_schedule`)


## Using our model
Our fine-tuned models can be fund in the folder [zip_models](https://github.com/borchand/SecondYearProject/tree/main/zip_models). The models are named by their `save_name` and are compressed to zip files. In order to use the models run:
```bash
python3 unzip_models.py
```
This will create a folder called models. Each model will be named as `xlm-mlm-17-1280-finetuned-ner-{save-name}`. To run a specific model, the `save_name` needs to be set in the `TokenClassificationTrainer`.
