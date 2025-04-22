local_dir="./data/tokenizer"

modelscope download --dataset 'BAAI/IndustryCorpus2'\
    accommodation_catering_hotel/english/high/rank_00726.parquet\
    news_media/chinese/high/rank_00082.parquet\
    news_media/english/high/rank_01332.parquet\
    mathematics_statistics/english/high/rank_01082.parquet\
    computer_programming_code/english/high/rank_00864.parquet\
    literature_emotion/chinese/high/rank_00063.parquet\
    --local_dir "${local_dir}"