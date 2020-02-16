import nltk
import csv
from nltk.stem.snowball import SnowballStemmer

# Selecting seven books from Gutenberg digital books and seven different authors

book_list = ["austen-emma.txt", "bible-kjv.txt",
             "bryant-stories.txt", "chesterton-ball.txt",
             "edgeworth-parents.txt", "melville-moby_dick.txt",
             "whitman-leaves.txt"]

# Adding genres to books manually

genre = ["romance", "religion", "stories", "fiction", "children", "adventure", "poetry"]


book_data = []
stemmer = SnowballStemmer('english')



# Defining get key function for sorted function
def getKey(item):
    return item[2]


# Adding items to book_data list as  tuples(Book Name, Author, Number of words, Genre)

i = 0

for book_name in book_list:
    t1 = (book_name, book_name.split("-")[0], len((nltk.corpus.gutenberg.raw(book_name)).split(" ")), genre[i])
    i = i + 1
    book_data.append(t1)


# Sorting the book_data list by number of words in a book

sorted(book_data, key=getKey)

# printing the book_data list

for book in book_data:
    print(book[0], book[1], book[2], book[3])

# Creating csv file

rows = []   # creating list for rows

for book in book_data:
    s = 0
    n = 150
    t1 = ()
    j = 1
    while j <= 200:
        x = (nltk.corpus.gutenberg.raw(book[0]).split()[s: s + n])  # taking 150 words at a time
        stemmed = [stemmer.stem(stem) for stem in x]
        sentence_with_150_words = " ".join(stemmed)             # sentence with 150 words
        t1 = (book[1], book[3], sentence_with_150_words)  # tuple (Author, Genre, Sentence)
        rows.append(t1)
        s = s + n
        j = j + 1


with open('data_set_200.csv', 'w') as out:
    csv_out = csv.writer(out)
    csv_out.writerow(['Author', 'Genre', 'Sentence'])
    csv_out.writerows(rows)



