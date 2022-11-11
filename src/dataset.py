# PyMed is a Python library that provides access to PubMed.
from pymed import PubMed


def get_articles_from_pubmed_database(mental_illness):
    # Create a PubMed object that GraphQL can use to query
    pubmed = PubMed(tool="sentiment-analysissss", email="strangerrrr@mail.com")

    # Create a GraphQL query
    query = mental_illness + "[Title]"

    # Execute the query against the PubMed API
    results = pubmed.query(query, max_results=5)
    articles = []
    for article in results:
        print(article.toJSON())
        articles.append(article)
    return articles


def get_article_content(article):
    return article.abstract


def filtering_invalid_content(content):
    return content is not None and len(content) > 10


def get_dataset():
    general_results = get_articles_from_pubmed_database("mental health")
    anxiety_results = get_articles_from_pubmed_database("anxiety")
    depression_results = get_articles_from_pubmed_database("depression")
    suicide_results = get_articles_from_pubmed_database("suicide")
    insomnia_results = get_articles_from_pubmed_database("insomnia")
    stress_results = get_articles_from_pubmed_database("stress")
    phobias_results = get_articles_from_pubmed_database("phobias")
    schizophrenia_results = get_articles_from_pubmed_database("schizophrenia")
    data = general_results + anxiety_results + depression_results + suicide_results + insomnia_results + stress_results + phobias_results + schizophrenia_results
    return list(filter(filtering_invalid_content, list(map(get_article_content, data))))
