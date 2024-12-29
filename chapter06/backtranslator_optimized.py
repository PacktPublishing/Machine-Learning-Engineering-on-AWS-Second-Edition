import torch
import pandas as pd
import argparse
import os
import importlib
import ast
import logging
import sys
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging_level = logging.getLevelName('INFO')
logging_stream_handler = logging.StreamHandler(sys.stdout)
logging_format = "%(asctime)s - %(message)s"
logging.basicConfig(
    level=logging_level,
    handlers=[logging_stream_handler],
    format=logging_format,
)
logger = logging.getLogger(__name__)


def log(log_string, logger=logger):
    logger.info(log_string)


def log_line(logger=logger):
    logger.info("█" * 50)
    

def get_tokenizer_and_model(model_name, device=DEVICE):
    log("get_tokenizer_and_model()")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    return (tokenizer, model)


def translate_list(statement_list, tokenizer, model, device=DEVICE):
    log("translate_list()")
    encoding = tokenizer(statement_list, return_tensors="pt", truncation=True, padding="longest").to(device)
    output = model.generate(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask)
    translated_text_list = [tokenizer.decode(seq, skip_special_tokens=True) for seq in output]
    
    return translated_text_list


def back_translate_list(statement_list, first_tokenizer, first_model, second_tokenizer, second_model, device=DEVICE):
    log("back_translate_list()")
    translated_text_list = translate_list(
        statement_list=statement_list, 
        tokenizer=first_tokenizer, 
        model=first_model, 
        device=device
    )
    output_list = translate_list(
        statement_list=translated_text_list, 
        tokenizer=second_tokenizer, 
        model=second_model, 
        device=device
    )
    return output_list


def generate_new_statement_list(statement_list, first_lang="tl", second_lang="en", device=DEVICE):
    log("generate_new_statement()")
    log(f"first_lang: { first_lang }")
    log(f"second_lang: { first_lang }")
    log(f"statement_list: { str(statement_list) }")
    
    first_model_name = f"Helsinki-NLP/opus-mt-{first_lang}-{second_lang}"
    second_model_name = f"Helsinki-NLP/opus-mt-{second_lang}-{first_lang}"
    
    first_tokenizer, first_model = get_tokenizer_and_model(first_model_name)
    second_tokenizer, second_model = get_tokenizer_and_model(second_model_name)
    
    new_statement_list = back_translate_list(
        statement_list=statement_list,
        first_tokenizer=first_tokenizer,
        first_model=first_model,
        second_tokenizer=second_tokenizer,
        second_model=second_model
    )
    
    log(f"new_statement_list: { str(new_statement_list) }")
    
    return new_statement_list


def process_args():
    log("process_args()")
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_language', type=str, default="tl")
    parser.add_argument('--batch_size', type=str, default='100')
    parser.add_argument('--languages_for_back_translation', type=str, default='["en","pt","de"]')
    arguments, _ = parser.parse_known_args()
    
    return arguments


def load_input(input_target):
    log("load_input()")
    
    log_line()
    df = pd.read_csv(input_target, header=None)
    print(df)
    log_line()
    
    return df


def deduplicate(list_of_statements):
    log("deduplicate()")
    return list(set(list_of_statements))


def main():
    log_line()
    log("[START]")
    log_line()
        
    log_line()
    log("[PROCESSING ARGS]")
    log_line()
    args = process_args()
    log_line()
    log(f"args.source_language = {args.source_language}")
    log(f"args.languages_for_back_translation = {args.languages_for_back_translation}")
    log_line()
    
    first_lang = args.source_language
    second_langs = ast.literal_eval(args.languages_for_back_translation)
    
    log_line()
    log("[LOADING INPUT CSV FILE]")
    log_line()
    df_of_statements = load_input("/opt/ml/processing/input/input.csv")
    log_line()
    total_rows = len(df_of_statements)
    log(f"original statements count: {total_rows}")
    log_line()
    
    log_line()
    log("[GENERATE NEW STATEMENTS (BACK TRANSLATION)]")
    log_line()
    
    batch_size = int(args.batch_size)
    batches = np.array_split(df_of_statements, len(df_of_statements) // batch_size)
    new_statements = []
    total_batches = len(batches)
    
    for index, batch in enumerate(batches):
        log_line()
        log(f"PROCESSING BATCH {index + 1} of {total_batches} ({batch_size} per batch)")
        log_line()
        statement_list = batch[0].to_list()
        for second_lang in second_langs:
            new_statement_list = generate_new_statement_list(statement_list, first_lang=first_lang, second_lang=second_lang)
            new_statements += new_statement_list
    
    log_line()
    print(f"generated statements count: {len(new_statements)}")
    unique_list = deduplicate(new_statements)
    print(f"unique generated statements count: {len(unique_list)}")
    log_line()
    
    log_line()
    log("[SAVING OUTPUT TO A CSV FILE]")
    log_line()
    df = pd.DataFrame(unique_list)
    df.to_csv("/opt/ml/processing/output/output.csv", index=False, header=False)
    
    log_line()
    log("[END]")
    log_line()
    
    
if __name__ == "__main__":
    main()
