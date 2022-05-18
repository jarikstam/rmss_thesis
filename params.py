sql_db = './data/film_discussions'
discussionarchive_submissions_pushshift = 'data/discussionarchive_submissions_pushshift.txt'

count_file = 'data/keywords_daily_count.csv'
submissions_tmdb_file = './data/submissions_tmdb.csv'
annotated_submissions_file = './data/submissions_annotated.csv'
comments_file = './data/comments.csv'
politics_comments_file = 'data/politics_comments.csv'
analysis_dataset = 'data/comments_analysis_v3.csv' # dataset which combines select twitter data, tmdb data

# Tokenized comments for matching discourse atoms and concept mover's distance to comments
tokenized_comments_text_file = './data/tokenized_comments.txt'

# Files imported for furhter processing
tokenized_sents_file = 'data/tokenized_sents.txt'

# Folder to save word embedding models
embedding_results_file = "./data/embedding_results.csv"

# Convert the chosen wv model to a dataframe and to binary. Dataframe used for CMD in R, binary used for R word2vec library
chosen_wv_model = './models/gensim_model_window25_vector300'
wvdf_file = 'data/wvdf.csv'
binary_wv_file = "./models/gensim_model_window25_vector300_kv.w2v"

reddit_praw_id = "Jarik"
