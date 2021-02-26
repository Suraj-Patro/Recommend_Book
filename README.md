# Book Recommendations from Charles Darwin
Build a book recommendation system using NLP and the text of books like "On the Origin of Species".

# Description
Recommendation systems are at the heart of many products such as Netflix or Amazon. They generally rely on metadata (e.g., the actors or director of a movie) or on user tastes (e.g., the movies you liked before) to determine which you are most likely to enjoy. But when you are working with text-heavy datasets, you have access to a much richer resource—the whole text! In this project, you will learn how to build the basis of a book recommendation system based on their content. You will use Charles Darwin's bibliography to find out which books might interest you.

The dataset was manually collected from [Project Gutenberg](https://www.gutenberg.org/).

# Tasks:
1. Darwin's bibliography
2. Load the contents of each book into Python
3. Find "On the Origin of Species"
4. Tokenize the corpus
5. Stemming of the tokenized corpus
6. Building a bag-of-words model
7. The most common words of a given book
8. Build a tf-idf model
9. The results of the tf-idf model
10. Compute distance between texts
11. The book most similar to "On the Origin of Species"
12. Which books have similar content?

# Topics Covered:
- Data Manipulation
- Data Visualization
- Probability & Statistics
- Importing & Cleaning Data

# Demonstration:

- *An executable or viewable **Jupyter Notebook**:* 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Suraj-Patro/Recommend_Book/blob/main/Recommend_Book.ipynb)
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/Suraj-Patro/Recommend_Book/blob/main/Recommend_Book.ipynb)
[![mybinder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Suraj-Patro/Recommend_Book/main?filepath=Recommend_Book.ipynb)

- *Executing [Recommend_Book.py](https://raw.githubusercontent.com/Suraj-Patro/Recommend_Book/main/recommend_book.py) in a Python Environment with the [required packages](#packages-required) installed.*

# Packages Required
- [glob](https://docs.python.org/3/library/glob.html)
- [RegEx](https://docs.python.org/3/library/re.html)
- [OS](https://docs.python.org/3/library/os.html)
- [Pickle](https://docs.python.org/3/library/pickle.html)
- [NLTK](https://www.nltk.org/index.html)
- [Gensim](https://radimrehurek.com/gensim/)
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/index.html#getting-started)
- [Matplotlib](https://matplotlib.org/stable/index.html)
- [SciPy](https://www.scipy.org/)
