import pandas as pd
import wikipediaapi


"""
From the movies.csv create a processed.csv file with the following columns: year, title, plot
"""


def get_plot(movie, year="1995"):
    wiki_wiki = wikipediaapi.Wikipedia("en")
    page_py = wiki_wiki.page(movie)
    if page_py.exists():
        try:
            return str(
                page_py.section_by_title("Plot").text[:500]
            )  # assure the output is a string
        except:
            return pd.NaT
    else:
        return pd.NaT


"""     else:
        page_py = wiki_wiki.page(movie + ' (' + year + ' film)')
        if page_py.exists():
            try:
                print(movie)
                return page_py.section_by_title('Plot').text[:500]
            except:
                print(movie)
                return pd.NaT """


def process_data(file):
    data = pd.read_csv(file, sep=",", names=["index", "prompt"])
    data = data[:500]
    data.drop("index", axis=1, inplace=True)  # drop the index column
    data["year"] = data["prompt"].str.extract(
        "(\d{4})", expand=True
    )  # extract the year
    data["title"] = data["prompt"].map(lambda x: x.split("(")[0])  # delete the name
    data["description"] = data["title"].map(lambda x: get_plot(x))  # get the plot
    data.drop("prompt", axis=1, inplace=True)
    data["text"] = data["title"] + " : " + data["description"]
    data.drop("title", axis=1, inplace=True)
    data.drop("description", axis=1, inplace=True)
    data.dropna(inplace=True)  # drop the rows with no plot
    data.to_csv("../data/processed.csv", sep=",", index=False)


if __name__ == "__main__":
    process_data("../data/movies.csv")
