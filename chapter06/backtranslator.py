import torch
import pandas as pd
import argparse
import os
import importlib
import ast
import logging
import sys

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


def log_section(log_string, logger=logger):
    log_line(logger=logger)
    log(log_string=log_string, logger=logger)
    log_line(logger=logger)
    

def get_tokenizer_and_model(model_name, device=DEVICE):
    log("get_tokenizer_and_model()")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    return (tokenizer, model)


def translate(statement, tokenizer, model, device=DEVICE):
    log("translate()")
    encoding = tokenizer(statement, return_tensors="pt", truncation=True, padding="longest").to(device)    
    output = model.generate(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask)
    translated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return translated_text


def back_translate(statement, first_tokenizer, first_model, second_tokenizer, second_model, device=DEVICE):
    log("back_translate()")
    translated_text = translate(statement=statement, tokenizer=first_tokenizer, model=first_model, device=device)
    output = translate(statement=translated_text, tokenizer=second_tokenizer, model=second_model, device=device)
    return output


def generate_new_statement(statement, first_lang="tl", second_lang="en", device=DEVICE):
    log("generate_new_statement()")
    log(f"first_lang: { first_lang }")
    log(f"second_lang: { first_lang }")
    log(f"statement: { statement }")
    
    first_model_name = f"Helsinki-NLP/opus-mt-{first_lang}-{second_lang}"
    second_model_name = f"Helsinki-NLP/opus-mt-{second_lang}-{first_lang}"
    
    first_tokenizer, first_model = get_tokenizer_and_model(first_model_name)
    second_tokenizer, second_model = get_tokenizer_and_model(second_model_name)
    
    new_statement = back_translate(
        statement=statement,
        first_tokenizer=first_tokenizer,
        first_model=first_model,
        second_tokenizer=second_tokenizer,
        second_model=second_model
    )
    
    log(f"new statement: { new_statement }")
    
    return new_statement


def process_args():
    log("process_args()")
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_language', type=str, default="tl")
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
    log_section("[START]")
    
    log_section("[PROCESSING ARGS]")
    
    args = process_args()
    log_line()
    log(f"args.source_language = {args.source_language}")
    log(f"args.languages_for_back_translation = {args.languages_for_back_translation}")
    log_line()
    
    first_lang = args.source_language
    second_langs = ast.literal_eval(args.languages_for_back_translation)
    
    log_section("[LOADING INPUT CSV FILE]")
    df_of_statements = load_input("/opt/ml/processing/input/input.csv")
    log_line()
    total_rows = len(df_of_statements)
    log(f"original statements count: {total_rows}")
    log_line()
    
    log_section("[GENERATE NEW STATEMENTS (BACK TRANSLATION)]")
    
    new_statements = []
    for index, row in df_of_statements.iterrows():
        log_line()
        log(f"PROCESSING {index + 1} of {total_rows}")
        log_line()
        statement = row[0]
        for second_lang in second_langs:
            new_statement = generate_new_statement(statement, first_lang=first_lang, second_lang=second_lang)
            new_statements.append(new_statement)
    
    log_line()
    print(f"generated statements count: {len(new_statements)}")
    unique_list = deduplicate(new_statements)
    print(f"unique generated statements count: {len(unique_list)}")
    log_line()
    
    log_section("[SAVING OUTPUT TO A CSV FILE]")
    df = pd.DataFrame(unique_list)
    df.to_csv("/opt/ml/processing/output/output.csv", index=False, header=False)
    
    log_section("[END]")
    
    
if __name__ == "__main__":
    main()
