import pandas as pd
import wikipediaapi
from transformers import pipeline
from tqdm import tqdm


"""
From the movies
"""


def summarize(text):
    # model = "t5-small"
    model = "philschmid/bart-large-cnn-samsum"
    summarizer = pipeline("summarization", model=model)
    summary_text = summarizer(text, max_length=300, min_length=250, do_sample=False)[0][
        "summary_text"
    ]
    return summary_text


def get_plot(movie, year="1995"):
    wiki_wiki = wikipediaapi.Wikipedia("en")
    wiki_name = [
        movie,
        movie + "(" + year + " film)",
        movie + " (film)",
    ]
    page_py = wiki_wiki.page(movie)
    for name in wiki_name:
        page_py = wiki_wiki.page(name)
        if page_py.exists():
            try:
                return summarize(str(page_py.section_by_title("Plot")))
            except:
                return pd.NaT


def process_data(file):
    data = pd.read_csv(file, sep=",", names=["index", "prompt"])
    data = data[:3000]
    data.drop("index", axis=1, inplace=True)  # drop the index column
    data["year"] = data["prompt"].str.extract(
        "(\d{4})", expand=True
    )  # extract the year
    data["title"] = data["prompt"].progress_map(
        lambda x: x.split("(")[0]
    )  # delete the name
    # get the plot
    data["description"] = data.progress_apply(
        lambda x: get_plot(movie=x["title"], year=x["year"]), axis=1
    )  # get the plot
    data.drop("prompt", axis=1, inplace=True)
    data["text"] = data["title"] + " : " + data["description"]
    data.drop("title", axis=1, inplace=True)
    data.drop("description", axis=1, inplace=True)
    data.dropna(inplace=True)  # drop the rows with no plot
    data.to_csv("../data/summarized_movies.csv", sep=",", index=False)


if __name__ == "__main__":
    tqdm.pandas()
    process_data("../data/movies.csv")
