from transformers import BertTokenizer, EncoderDecoderModel

# Carregar o tokenizer e o modelo treinado
model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(r"D:\Projetos\Pos\BERT\drive\bert-tokenizer-translation-en-pt-v3")
model = EncoderDecoderModel.from_pretrained(r"D:\Projetos\Pos\BERT\drive\bert-translation-en-pt-v3")

# Definir o decoder_start_token_id e pad_token_id
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id

# Função de tradução
def translate(text, max_length=50):
    # Codifique o texto
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # Defina o bos_token_id e o decoder_start_token_id
    bos_token_id = tokenizer.bos_token_id
    decoder_start_token_id = tokenizer.cls_token_id

    # Prepare as entradas para geração
    inputs = {
        "input_ids": input_ids,
        "decoder_start_token_id": decoder_start_token_id,
    }

    # Gere a saída
    outputs = model.generate(**inputs, max_length=100, num_return_sequences=1, do_sample=True, temperature=0.3)

    # Decodifique a saída gerada
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

def main():
    print(f"decoder_start_token_id: {model.config.decoder_start_token_id}", tokenizer.cls_token_id)
    print(f"pad_token_id: {model.config.pad_token_id}", tokenizer.pad_token_id)
    print(f"bos_token_id: {model.config.bos_token_id}", tokenizer.cls_token_id)
    print(f"eos_token_id: {model.config.eos_token_id}", tokenizer.sep_token_id)

    # Texto de exemplo para tradução
    text = "Good morning"

    print(f"Texto original: {text}")
    print(f"Texto traduzido: {translate(text)}")

if __name__ == '__main__':
    main()