import sqlite3
import requests
import json
import unicodedata
from nltk import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer
import regex as re
import string


def get_submission_ids(conn, table, column="submission_id"):
    """
    Return a set of all values in a column from a SQL table
    Based on: https://www.sqlitetutorial.net/sqlite-python/sqlite-python-select/
    :param conn: the Connection object
    :param table: an SQL table
    :param column: the name of a column
    :return: a set of values
    """
    cur = conn.cursor()
    cur.execute(f"SELECT {column} FROM {table}")

    # fetchall returns a list of tuples of values
    # Which I convert to a set of values
    submission_ids = set((submission_id[0] for submission_id in cur.fetchall()))

    return submission_ids

def get_pushshift_data(param_dict, url='https://api.pushshift.io/reddit/search/submission/?', data_only=True):
    """
    Return data from the pushshift API
    Based on: https://github.com/SeyiAgboola/Reddit-Data-Mining/blob/master/Using_Pushshift_Module_to_extract_Submissions.ipynb
    :param param_dict: A dictionary with key+value pairs to feed to the API
    :param url: The URL of the pushshift API
    :return: A json object
    """
    for k, v in param_dict.items():
        url = f'{url}{k}={v}&'

    url = url[:-1]
    r = requests.get(url)
    assert r.status_code == 200, r.status_code
    data = r.json()

    if data_only:
        return data['data']
    else:
        return data

def collect_submission_data(submission, keys=('id', 'title', 'score', 'num_comments', 'url', 'created_utc')):
    """
    Collect selected data from a dictionary or .json object
    Based on: https://github.com/SeyiAgboola/Reddit-Data-Mining/blob/master/Using_Pushshift_Module_to_extract_Submissions.ipynb
    :param submission: A json object from the pushshift API
    :param submission_keys: An iterable matching to keys in the submission
    :return: A list with submission data
    """
    """submission_id = submission['id']
    title = submission['title']
    score = submission['score']
    num_comments = submission['num_comments']
    url = submission['url']
    created = submission['created_utc']
    submission_data = (submission_id, title, score, num_comments, url, created)"""

    submission_data = [submission[key] for key in keys]
    return submission_data

def get_comments_data(reddit, submission_id, comment_attributes=('comment.id', 'submission.id', 'comment.body', 'str(comment.author)', 'comment.stickied', 'comment.score', 'comment.created_utc')):
    """
    Collect all the comments from a Reddit post
    :param reddit: a praw.Reddit instance
    :param submission_id: the ID of a Reddit submission
    :param comment_attributes: an iterable of comment attributes
    :return: comments data
    """
    submission = reddit.submission(submission_id)

    # Collect all comments from the submission
    submission.comments.replace_more(limit=None)

    # eval turns the string into a functioning attribue, the dictionary is needed to tell eval that 'comment' is coming from the global environment
    # This old line perhaps caused errors for deleted comments, thus the new version.
    #comments_data = [[eval(comment_attribute, {"comment":comment, "submission":submission}) for comment_attribute in comment_attributes] for comment in submission.comments.list()]
    comments_data = []
    for comment in submission.comments.list():
        if comment.author is not None:
            comment_data = [eval(comment_attribute, {"comment":comment, "submission":submission}) for comment_attribute in comment_attributes]
            comments_data.append(comment_data)

    return comments_data

def add_rows(conn, table_name, n_columns, data_rows):
    """
    Add rows to a SQL table
    :param conn: Connection object
    :param table_name: name of SQL table
    :param n_columns: number of columns in table
    :param data_rows: a list of rows to be added to table
    :return: The number of rows added
    """
    vals='?,'*n_columns
    sql = f''' REPLACE INTO {table_name}
              VALUES({vals[:-1]}) '''
    # sql = f''' INSERT OR IGNORE INTO {table_name}
    #          VALUES({vals[:-1]}) '''
    cur = conn.cursor()
    cur.executemany(sql, data_rows)
    conn.commit()
    return len(data_rows)

def interact_with_db(conn, execute_sql, fetch=None, commit=False):
    """
    Execute something in SQL, potentially fetch
    :param conn: A connection object
    :param execute_sql: An sql execute statement
    :param fetch: A string with a complete fetch statement
    :param commit: Boolean to commit
    :return: if fetch!=None returns the fetched object
    """
    cur = conn.cursor()
    cur.execute(execute_sql)
    if fetch != None:
        x = eval(fetch)
        return x
    if commit == True:
        conn.commit()

def strip_accents(text):
    """
    Takes and returns a string where the diacritics (such as accents) from letter
    characters in that string are replaced with their basic (unaccented) versions.
    Copied from: https://stackoverflow.com/questions/44431730/how-to-replace-accented-characters-in-python/44433664#44433664
    The only change I made is that I return 'text' instead of 'str(text)', since it's already a string as one of the comments points out.
    :param text: A string
    :return: A string
    """
    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3
        pass

    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

    return text

def sentenize_comments(comments):
    """
    Takes an iterable of texts and returns a set of unique sentences in those texts
    :param comments: a list of strings
    :return: a set of unique sentences
    """
    unique_sentences = set()
    for comment in comments:
        comment = comment.lower()
        comment = strip_accents(comment)
        for sent in sent_tokenize(comment):
            sent = sent.strip()
            if sent not in unique_sentences:
                unique_sentences.add(sent)
    return unique_sentences

def tokenize_sentence(sentence, punctuation=set(string.punctuation)):
    """
    Tokenize a sentence into a list of strings, removing any tokens that are merely punctuation
    :param sentence: A string
    :param punctuation: a list of punctuation marks
    :return: a list of tokens
    """
    tokens = TreebankWordTokenizer().tokenize(sentence)
    tokens = [t for t in tokens if not all(j in punctuation for j in t)]
    return tokens
