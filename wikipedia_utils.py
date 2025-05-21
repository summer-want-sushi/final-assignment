import wikipedia
import spacy


def get_wiki_query(query):
    try:
        ### spacy code

        # Load the English model
        nlp = spacy.load("en_core_web_sm")

        # Parse the sentence
        doc = nlp(query)

        # Entity path (people, evenrs, books)
        entities_components = [entity_substring.text for entity_substring in doc.ents]
        if len(entities_components) > 0:
            subject_of_the_query = ""
            for substrings in entities_components:
                subject_of_the_query = subject_of_the_query + substrings

            if subject_of_the_query == "":
                print("Entity query not parsed.")
            return subject_of_the_query



        else:
            first_noun = next((t for t in doc if t.pos_ in {"NOUN", "PROPN"}), None).text
            print("Returning first noun from the query.")
            return first_noun




    except Exception as e:
        print("Failed parsing a query subject from query", query)
        print(e)


def fetch_wikipedia_page(wiki_query):
    try:
        matched_articles = wikipedia.search(wiki_query)
        if len(matched_articles) > 0:
            used_article = matched_articles[0]
            page_content = wikipedia.page(used_article, auto_suggest=False)
            return page_content.content
        else:
            return ""
    except Exception as e:
        print("Could not fetch the wikipedia article using ", wiki_query)
        print(e)
