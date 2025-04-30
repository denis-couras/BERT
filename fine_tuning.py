from datasets import load_dataset
from transformers import BertTokenizer, EncoderDecoderModel, Trainer, TrainingArguments
import torch
from transformers import get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from accelerate import Accelerator
from torch.utils.data import DataLoader

# Inicializar Accelerator
accelerator = Accelerator(mixed_precision='fp16')

# Carregar o dataset
all_dataset = load_dataset("VanessaSchenkel/translation-en-pt", field="data")

dataset = all_dataset['train'].shuffle(seed=42).select(range(int(len(all_dataset['train']) * 0.01)))
model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)

# Função para preparar os dados no formato correto
def preprocess_function(examples):
    # Extrair inputs e targets
    inputs = [ex["portuguese"] for ex in examples["translation"]]
    targets = [ex["english"] for ex in examples["translation"]]

    # Tokenizar inputs
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True)

    # Tokenizar targets e process labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, padding="max_length", truncation=True)["input_ids"]

    # Ensure labels match the input length and shape
    model_inputs["labels"] = labels

    return model_inputs

def main():
    # Carregar o tokenizer e o modelo BERT
    
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)

    # Definir o decoder_start_token_id
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Tokenizar o dataset
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    train_dataloader = DataLoader(tokenized_datasets, batch_size=16, shuffle=True, num_workers=4)

    # Configurar os parâmetros de treinamento com ajuste adicional
    training_args = TrainingArguments(
        output_dir=r"D:\Projetos\Pos\BERT\results",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=200,
        eval_steps=200,
        fp16=True,
        gradient_accumulation_steps=1
    )

    # Inicializar o Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets
    )

    torch.cuda.is_available()

    # Treinar o modelo
    trainer.train()

    # Salvar o modelo
    trainer.save_model(r"D:\Projetos\Pos\BERT\drive\bert-translation-en-pt-v3")
    tokenizer.save_pretrained(r"D:\Projetos\Pos\BERT\drive\bert-tokenizer-translation-en-pt-v3")

if __name__ == '__main__':
    main()