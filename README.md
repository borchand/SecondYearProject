# SecondYearProject
The following project is created in connection with our Second Year Project "Optimizing for cross-lingual learning for multilingual language models on unseen languages of similar structures".
The goal of our project is to answer the following research question: __How can model adaptation be adjusted to better utilizethe cross-domain knowledge obtained by multilingual LLM’s during pre-training and does these changes impact transferability to unseen
languages?__.

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
    Example:
    ```bash
    python3 train.py --discriminative_lr True
    ```

Other hyperparameters can be used by using the following arguments:
- batch_size: `--batch_size`
- learning_rate: `--lr`
- epochs: `--epochs`
- seed: `--seed`


## Recreating significance testing of models
    All results will be saved to the eval_lists folder
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

    Example:
    ```bash
    python3 train.py --discriminative_lr True --cosine_schedule True
    ```
Other hyperparameters can be used by using the following arguments:
- batch_size: `--batch_size`
- to_csv: `--to_csv` (default=True)
- save_name: `--save_name` (This will override `--discriminative_lr` and `--cosine_schedule`)


# Seed from our models
## Baseline
- 94664
- 16538
- 36677
- 39377
- 85712
- 99578
- 78252
- 97696
- 77020
- 79002
## Discriminate learning rate
- 36916
- 32320
- 22986
- 66448
- 68125
- 3837
- 9756
- 3168
- 70121
- 57808
