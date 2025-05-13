import json
from datetime import datetime

from InfoRetriever.eval_interface.interface import Index, SearchEngine
import subprocess

from timeit import timeit, default_timer as timer


def load_queries(file_path):
    queries = []
    with open(file_path, 'r') as f:
        for line in f:
            query = line.strip()
            queries.append(query)
    return queries

def load_data_json(path):
    """
    Load documents from a json file
    :param path:
    :return:
    """
    with open(path, 'r') as f:
        return json.load(f)

def dump_data_json(data, path):
    """
    Dump data to a json file
    :param data:
    :param path:
    :return:
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def run_trec_eval(gold_file, results_file):
    # Run TREC eval
    p = subprocess.Popen(['trec_eval-main/trec_eval', '-m', 'all_trec', gold_file, results_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    out = out.decode('utf-8')
    err = err.decode('utf-8')
    print(out)
    print(err)


def evaluate_ranked(search_engine, queries, language='en'):
    results_file = f"../output/ranked_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    with open(results_file, "w") as f:
        for query in queries:
            results = search_engine.search(query["description"])
            for i, (doc_id, score) in enumerate(results):
                f.write(f"{query['id']} Q0 {doc_id} {i+1} {score} runindex1\n")
    run_trec_eval(f'../data/{language}/gold_relevancies.txt', results_file)

def evaluate_boolean(search_engine, queries):
    for query_id, query in enumerate(queries):
        results = search_engine.boolean_search(query)
        assert len(results) > 0
        print(len(results))



if __name__ == '__main__':
    language = 'cs'
    documents = load_data_json(f'../data/{language}/documents.json')
    ranked_queries = load_data_json(f'../data/{language}/full_text_queries.json')
    boolean_queries = load_queries(f'../data/{language}/boolean_queries_standard_100.txt')
    index = Index()
    print(f"Indexing took {timeit(lambda :index.index_documents(documents), number=1)} seconds.")
    search_engine = SearchEngine(index)
    print(f"Ranked evaluation took {timeit(lambda :evaluate_ranked(search_engine, ranked_queries, language), number=1)} seconds.")
    print(f"Boolean evaluation took {timeit(lambda :evaluate_boolean(search_engine, boolean_queries), number=1)} seconds.")
