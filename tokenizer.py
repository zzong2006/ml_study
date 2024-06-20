from pprint import pformat

from loguru import logger
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq


target_model = "Qwen/Qwen2-72B"
tokenizer = AutoTokenizer.from_pretrained(target_model)


def fetchSamsumDataset(sample_size: int = 5):
    dataset_samsum = load_dataset("samsum", split=["train"], trust_remote_code=True)[0].take(sample_size)
    logger.info("Dataset: {}", pformat(dataset_samsum))
    return dataset_samsum


def testContextTokenization(sample_size: int = 5):
    dataset_samsum = fetchSamsumDataset(sample_size)
    target_batch = dataset_samsum["summary"]

    with tokenizer.as_target_tokenizer():       # Actually there is no difference
        target_tensor = tokenizer(target_batch, max_length=10, truncation=True)
        logger.info("Tokenized: {}", pformat(target_tensor))

    result_without_ctx = tokenizer(target_batch, max_length=10, truncation=True)
    logger.info("Tokenized without context: {}", pformat(result_without_ctx))


if __name__ == "__main__":
    testContextTokenization()