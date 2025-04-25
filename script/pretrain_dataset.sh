local_dir="./data/pretrain"

modelscope download --dataset 'BAAI/IndustryCorpus2'\
    --include\
    'artificial_intelligence_machine_learning/chinese/high/*.parquet'\
    'artificial_intelligence_machine_learning/english/high/*.parquet'\
    'computer_programming_code/chinese/high/*.parquet'\
    'computer_programming_code/english/high/*.parquet'\
    'mathematics_statistics/chinese/high/*.parquet'\
    'mathematics_statistics/english/high/rank_0108*.parquet'\
    'accommodation_catering_hotel/chinese/high/*.parquet'\
    'accommodation_catering_hotel/english/high/*.parquet'\
    'biomedicine/chinese/high/*.parquet'\
    'biomedicine/english/high/rank_00793.parquet' 'biomedicine/english/high/rank_00794.parquet' 'biomedicine/english/high/rank_00795.parquet' 'biomedicine/english/high/rank_00796.parquet'\
    'computer_communication/chinese/high/rank_00017.parquet'\
    'computer_communication/english/high/rank_0083*.parquet'\
    'news_media/chinese/high/*.parquet'\
    'news_media/english/high/*.parquet'\
    'tourism_geography/chinese/high/rank_00128.parquet'\
    'tourism_geography/english/high/rank_01529.parquet' 'tourism_geography/english/high/rank_01530.parquet'\
    'technology_scientific_research/chinese/high/rank_00123.parquet'\
    'technology_scientific_research/english/high/rank_0146*.parquet'\
    'film_entertainment/chinese/high/rank_00042.parquet'\
    'film_entertainment/english/high/rank_00961.parquet' 'film_entertainment/english/high/rank_00962.parquet'\
    'current_affairs_government_administration/chinese/high/rank_00026.parquet' 'current_affairs_government_administration/chinese/high/rank_00027.parquet'\
    'current_affairs_government_administration/english/high/rank_0086*.parquet'\
    'game/chinese/high/*.parquet'\
    'game/english/high/*.parquet'\
    --local_dir "${local_dir}"