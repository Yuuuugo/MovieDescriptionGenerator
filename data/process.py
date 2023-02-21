import pandas as pd
import wikipediaapi

def get_plot(movie):
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page_py = wiki_wiki.page(movie)
    if page_py.exists():
        return page_py.section_by_title('Plot').text
    else:
        return None



def process_data(file):
    data = pd.read_csv(file, sep=',', names = ['index', 'movie'])
    data = data[:5]
    data.drop('index', axis=1, inplace=True) # drop the index column
    data['movie'] = data['movie'].map(lambda x: x.split("(")[0]) # delete the name
    data['plot'] = data['movie'].map(lambda x: get_plot(x)) # get the plot
    print(data)

if __name__ == '__main__':
    process_data('movies.csv')
