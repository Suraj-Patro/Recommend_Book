*This page is available as an executable or viewable **Jupyter Notebook**:* 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Suraj-Patro/Recommend_Book/blob/main/Recommend_Book.ipynb)
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/Suraj-Patro/Recommend_Book/blob/main/Recommend_Book.ipynb)
[![mybinder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Suraj-Patro/Recommend_Book/main?filepath=Recommend_Book.ipynb)

## 1. Darwin's bibliography
<p><img src="https://assets.datacamp.com/production/project_607/img/CharlesDarwin.jpg" alt="Charles Darwin" width="300px"></p>
<p>Charles Darwin is one of the few universal figures of science. His most renowned work is without a doubt his "<em>On the Origin of Species</em>" published in 1859 which introduced the concept of natural selection. But Darwin wrote many other books on a wide range of topics, including geology, plants or his personal life. In this notebook, we will automatically detect how closely related his books are to each other.</p>
<p>To this purpose, we will develop the bases of <strong>a content-based book recommendation system</strong>, which will determine which books are close to each other based on how similar the discussed topics are. The methods we will use are commonly used in text- or documents-heavy industries such as legal, tech or customer support to perform some common task such as text classification or handling search engine queries.</p>
<p>Let's take a look at the books we'll use in our recommendation system.</p>


```python
# Loading datasets
!wget -qnc -P datasets https://raw.githubusercontent.com/Suraj-Patro/Recommend_Book/main/datasets/Autobiography.txt
!wget -qnc -P datasets https://raw.githubusercontent.com/Suraj-Patro/Recommend_Book/main/datasets/CoralReefs.txt
!wget -qnc -P datasets https://raw.githubusercontent.com/Suraj-Patro/Recommend_Book/main/datasets/DescentofMan.txt
!wget -qnc -P datasets https://raw.githubusercontent.com/Suraj-Patro/Recommend_Book/main/datasets/DifferentFormsofFlowers.txt
!wget -qnc -P datasets https://raw.githubusercontent.com/Suraj-Patro/Recommend_Book/main/datasets/EffectsCrossSelfFertilization.txt
!wget -qnc -P datasets https://raw.githubusercontent.com/Suraj-Patro/Recommend_Book/main/datasets/ExpressionofEmotionManAnimals.txt
!wget -qnc -P datasets https://raw.githubusercontent.com/Suraj-Patro/Recommend_Book/main/datasets/FormationVegetableMould.txt
!wget -qnc -P datasets https://raw.githubusercontent.com/Suraj-Patro/Recommend_Book/main/datasets/FoundationsOriginofSpecies.txt
!wget -qnc -P datasets https://raw.githubusercontent.com/Suraj-Patro/Recommend_Book/main/datasets/GeologicalObservationsSouthAmerica.txt
!wget -qnc -P datasets https://raw.githubusercontent.com/Suraj-Patro/Recommend_Book/main/datasets/InsectivorousPlants.txt
!wget -qnc -P datasets https://raw.githubusercontent.com/Suraj-Patro/Recommend_Book/main/datasets/LifeandLettersVol1.txt
!wget -qnc -P datasets https://raw.githubusercontent.com/Suraj-Patro/Recommend_Book/main/datasets/LifeandLettersVol2.txt
!wget -qnc -P datasets https://raw.githubusercontent.com/Suraj-Patro/Recommend_Book/main/datasets/MonographCirripedia.txt
!wget -qnc -P datasets https://raw.githubusercontent.com/Suraj-Patro/Recommend_Book/main/datasets/MonographCirripediaVol2.txt
!wget -qnc -P datasets https://raw.githubusercontent.com/Suraj-Patro/Recommend_Book/main/datasets/MovementClimbingPlants.txt
!wget -qnc -P datasets https://raw.githubusercontent.com/Suraj-Patro/Recommend_Book/main/datasets/OriginofSpecies.txt
!wget -qnc -P datasets https://raw.githubusercontent.com/Suraj-Patro/Recommend_Book/main/datasets/PowerMovementPlants.txt
!wget -qnc -P datasets https://raw.githubusercontent.com/Suraj-Patro/Recommend_Book/main/datasets/VariationPlantsAnimalsDomestication.txt
!wget -qnc -P datasets https://raw.githubusercontent.com/Suraj-Patro/Recommend_Book/main/datasets/VolcanicIslands.txt
!wget -qnc -P datasets https://raw.githubusercontent.com/Suraj-Patro/Recommend_Book/main/datasets/VoyageBeagle.txt
!wget -qnc -P datasets https://raw.githubusercontent.com/Suraj-Patro/Recommend_Book/main/datasets/text_lem.p
!wget -qnc -P datasets https://raw.githubusercontent.com/Suraj-Patro/Recommend_Book/main/datasets/texts_stem.p
```


```python
# Importing modules
import glob

# The books files are contained in this folder
folder = "datasets/"

# List all the .txt files and sort them alphabetically
files = glob.glob(folder+"*.txt")
files.sort()
files
```




    ['datasets/Autobiography.txt',
     'datasets/CoralReefs.txt',
     'datasets/DescentofMan.txt',
     'datasets/DifferentFormsofFlowers.txt',
     'datasets/EffectsCrossSelfFertilization.txt',
     'datasets/ExpressionofEmotionManAnimals.txt',
     'datasets/FormationVegetableMould.txt',
     'datasets/FoundationsOriginofSpecies.txt',
     'datasets/GeologicalObservationsSouthAmerica.txt',
     'datasets/InsectivorousPlants.txt',
     'datasets/LifeandLettersVol1.txt',
     'datasets/LifeandLettersVol2.txt',
     'datasets/MonographCirripedia.txt',
     'datasets/MonographCirripediaVol2.txt',
     'datasets/MovementClimbingPlants.txt',
     'datasets/OriginofSpecies.txt',
     'datasets/PowerMovementPlants.txt',
     'datasets/VariationPlantsAnimalsDomestication.txt',
     'datasets/VolcanicIslands.txt',
     'datasets/VoyageBeagle.txt']



## 2. Load the contents of each book into Python
<p>As a first step, we need to load the content of these books into Python and do some basic pre-processing to facilitate the downstream analyses. We call such a collection of texts <strong>a corpus</strong>. We will also store the titles for these books for future reference and print their respective length to get a gauge for their contents.</p>


```python
# Import libraries
import re, os

# Initialize the object that will contain the texts and titles
txts = []
titles = []

for n in files:
  # Open each file
  f = open(n, encoding='utf-8-sig')
  # Remove all non-alpha-numeric characters
  data = re.sub('[\W_]+', ' ', f.read())
  # Store the texts and titles of the books in two separate lists
  txts.append(data)
  titles.append(os.path.basename(n).replace(".txt", ""))
    
# Print the length, in characters, of each book
[len(t) for t in txts]
```




    [123231,
     496068,
     1776539,
     617088,
     913713,
     624232,
     335920,
     523021,
     797401,
     901406,
     1047518,
     1010643,
     767492,
     1660866,
     298319,
     916267,
     1093567,
     1043499,
     341447,
     1149574]



## 3. Find "On the Origin of Species"
<p>For the next parts of this analysis, we will often check the results returned by our method for a given book. For consistency, we will refer to Darwin's most famous book: "<em>On the Origin of Species</em>." Let's find to which index this book is associated.</p>


```python
# Browse the list containing all the titles
for i in range(len(titles)):
  # Store the index if the title is "OriginofSpecies"
  if(titles[i]=="OriginofSpecies"):
    ori = i
 
# Print the stored index
print(str(ori))
```

    15


## 4. Tokenize the corpus
<p>As a next step, we need to transform the corpus into a format that is easier to deal with for the downstream analyses. We will tokenize our corpus, i.e., transform each text into a list of the individual words (called tokens) it is made of. To check the output of our process, we will print the first 20 tokens of "<em>On the Origin of Species</em>".</p>


```python
# Define a list of stop words
stoplist = set('for a of the and to in to be which some is at that we i who whom show via may my our might as well'.split())

# Convert the text to lower case 
txts_lower_case = [txt.lower() for txt in txts]

# Transform the text into tokens 
txts_split = [txt.split() for txt in txts_lower_case]

# Remove tokens which are part of the list of stop words
texts = [[word for word in txt if word not in stoplist] for txt in txts_split]

# Print the first 20 tokens for the "On the Origin of Species" book
texts[ori][0:20]
```




    ['on',
     'origin',
     'species',
     'but',
     'with',
     'regard',
     'material',
     'world',
     'can',
     'least',
     'go',
     'so',
     'far',
     'this',
     'can',
     'perceive',
     'events',
     'are',
     'brought',
     'about']



## 5. Stemming of the tokenized corpus
<p>If you have read <em>On the Origin of Species</em>, you will have noticed that Charles Darwin can use different words to refer to a similar concept. For example, the concept of selection can be described by words such as <em>selection</em>, <em>selective</em>, <em>select</em> or <em>selects</em>. This will dilute the weight given to this concept in the book and potentially bias the results of the analysis.</p>
<p>To solve this issue, it is a common practice to use a <strong>stemming process</strong>, which will group together the inflected forms of a word so they can be analysed as a single item: <strong>the stem</strong>. In our <em>On the Origin of Species</em> example, the words related to the concept of selection would be gathered under the <em>select</em> stem.</p>
<p>As we are analysing 20 full books, the stemming algorithm can take several minutes to run and, in order to make the process faster, we will directly load the final results from a pickle file and review the method used to generate it.</p>


```python
import pickle
#### Load the Porter stemming function from the nltk package
from nltk.stem import PorterStemmer

#### Create an instance of a PorterStemmer object
porter = PorterStemmer()

#### For each token of each text, we generated its stem 
texts_stem = [[porter.stem(token) for token in text] for text in texts]

#### Save to pickle file
pickle.dump( texts_stem, open( "texts_stem.p", "wb" ) )
```

## 6. Building a bag-of-words model
<p>Now that we have transformed the texts into stemmed tokens, we need to build models that will be useable by downstream algorithms.</p>
<p>First, we need to will create a universe of all words contained in our corpus of Charles Darwin's books, which we call <em>a dictionary</em>. Then, using the stemmed tokens and the dictionary, we will create <strong>bag-of-words models</strong> (BoW) of each of our texts. The BoW models will represent our books as a list of all uniques tokens they contain associated with their respective number of occurrences. </p>
<p>To better understand the structure of such a model, we will print the five first elements of one of the "<em>On the Origin of Species</em>" BoW model.</p>


```python
# Load the functions allowing to create and use dictionaries
from gensim import corpora

# Create a dictionary from the stemmed tokens
dictionary = corpora.Dictionary(texts_stem)

# Create a bag-of-words model for each book, using the previously generated dictionary
bows = [dictionary.doc2bow(text) for text in texts_stem]


# Print the first five elements of the On the Origin of species' BoW model
bows[ori][0:5]
```




    [(0, 11), (5, 51), (6, 1), (8, 2), (21, 1)]



## 7. The most common words of a given book
<p>The results returned by the bag-of-words model is certainly easy to use for a computer but hard to interpret for a human. It is not straightforward to understand which stemmed tokens are present in a given book from Charles Darwin, and how many occurrences we can find.</p>
<p>In order to better understand how the model has been generated and visualize its content, we will transform it into a DataFrame and display the 10 most common stems for the book "<em>On the Origin of Species</em>".</p>


```python
# Import pandas to create and manipulate DataFrames
import pandas as pd

# Convert the BoW model for "On the Origin of Species" into a DataFrame
df_bow_origin = pd.DataFrame(bows[ori])

# Add the column names to the DataFrame
df_bow_origin.columns = ["index", "occurrences"]

# Add a column containing the token corresponding to the dictionary index
df_bow_origin["token"] = [dictionary[index] for index in df_bow_origin["index"]]

# Sort the DataFrame by descending number of occurrences and print the first 10 values
df_bow_origin.sort_values(by="occurrences", ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>occurrences</th>
      <th>token</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>748</th>
      <td>1168</td>
      <td>2023</td>
      <td>have</td>
    </tr>
    <tr>
      <th>1119</th>
      <td>1736</td>
      <td>1558</td>
      <td>on</td>
    </tr>
    <tr>
      <th>1489</th>
      <td>2288</td>
      <td>1543</td>
      <td>speci</td>
    </tr>
    <tr>
      <th>892</th>
      <td>1366</td>
      <td>1480</td>
      <td>it</td>
    </tr>
    <tr>
      <th>239</th>
      <td>393</td>
      <td>1362</td>
      <td>by</td>
    </tr>
    <tr>
      <th>1128</th>
      <td>1747</td>
      <td>1201</td>
      <td>or</td>
    </tr>
    <tr>
      <th>125</th>
      <td>218</td>
      <td>1140</td>
      <td>are</td>
    </tr>
    <tr>
      <th>665</th>
      <td>1043</td>
      <td>1137</td>
      <td>from</td>
    </tr>
    <tr>
      <th>1774</th>
      <td>2703</td>
      <td>1000</td>
      <td>with</td>
    </tr>
    <tr>
      <th>1609</th>
      <td>2452</td>
      <td>962</td>
      <td>thi</td>
    </tr>
  </tbody>
</table>
</div>



## 8. Build a tf-idf model
<p>If it wasn't for the presence of the stem "<em>speci</em>", we would have a hard time to guess this BoW model comes from the <em>On the Origin of Species</em> book. The most recurring words are, apart from few exceptions, very common and unlikely to carry any information peculiar to the given book. We need to use an additional step in order to determine which tokens are the most specific to a book.</p>
<p>To do so, we will use a <strong>tf-idf model</strong> (term frequency–inverse document frequency). This model defines the importance of each word depending on how frequent it is in this text and how infrequent it is in all the other documents. As a result, a high tf-idf score for a word will indicate that this word is specific to this text.</p>
<p>After computing those scores, we will print the 10 words most specific to the "<em>On the Origin of Species</em>" book (i.e., the 10 words with the highest tf-idf score).</p>


```python
# Load the gensim functions that will allow us to generate tf-idf models
from gensim.models import TfidfModel

# Generate the tf-idf model
model = TfidfModel(bows)

# Print the model for "On the Origin of Species"
model[bows[ori]]
```




    [(8, 0.00020408683039616044),
     (21, 0.0005723177174474165),
     (23, 0.001714008058633542),
     (27, 0.0006466337090765656),
     (28, 0.0025710120879503125),
     (31, 0.000857004029316771),
     (35, 0.001016241827361025),
     (36, 0.001016241827361025),
     (51, 0.0008878482198272165),
     (54, 0.00203248365472205),
     (56, 0.0023786863377481767),
     (57, 0.00010204341519808022),
     (63, 0.0027579084706411254),
     (64, 0.000510217075990401),
     (66, 0.00020408683039616044),
     (67, 0.0023786863377481767),
     (68, 0.00203248365472205),
     (75, 0.0013789542353205627),
     (76, 0.00044392410991360827),
     (78, 0.004177054182752995),
     (80, 0.0020885270913764977),
     (83, 0.008584765761711247),
     (84, 0.000510217075990401),
     (88, 0.0024490419647539255),
     (89, 0.0033674327015366474),
     (90, 0.0008878482198272165),
     (91, 0.0016768424759030323),
     (94, 0.0008878482198272165),
     (95, 0.00044392410991360827),
     (96, 0.003551392879308866),
     (97, 0.0016326946431692835),
     (102, 0.03773354934265671),
     (104, 0.0009183907367827219),
     (106, 0.00141914571211187),
     (108, 0.0035478642802796744),
     (109, 0.005306257590300172),
     (111, 0.002244955134357765),
     (114, 0.0015306512279712034),
     (123, 0.05104059366655679),
     (125, 0.00959208102861954),
     (126, 0.004177054182752995),
     (127, 0.00141914571211187),
     (137, 0.020051661797575882),
     (139, 0.007759604508918787),
     (141, 0.001020434151980802),
     (143, 0.0047573726754963535),
     (144, 0.0047656786344253245),
     (154, 0.0010480265474393951),
     (156, 0.01397994788213699),
     (165, 0.006789653945303939),
     (167, 0.0006122604911884814),
     (172, 0.02179895218673942),
     (176, 0.0033674327015366474),
     (178, 0.0031074687693952584),
     (186, 0.0092859507830253),
     (188, 0.0016768424759030323),
     (192, 0.0062655812741294935),
     (196, 0.0038798022544593936),
     (197, 0.0013789542353205627),
     (198, 0.0008878482198272165),
     (204, 0.00425743713633561),
     (207, 0.0022632179817679795),
     (212, 0.001326564397575043),
     (214, 0.020420509056025982),
     (215, 0.021748073263001825),
     (219, 0.0019399011272296968),
     (220, 0.004401711499245459),
     (221, 0.0054686350357025125),
     (222, 0.0018228783452341711),
     (223, 0.000709572856055935),
     (224, 0.007713036263850938),
     (226, 0.0030613024559424068),
     (230, 0.005142024175900625),
     (231, 0.0027579084706411254),
     (235, 0.0024490419647539255),
     (236, 0.0012932674181531311),
     (237, 0.0020960530948787902),
     (241, 0.0029344743328303066),
     (242, 0.0015885595448084417),
     (243, 0.008273725411923376),
     (245, 0.004526435963535959),
     (246, 0.0034694761167347273),
     (247, 0.0017347380583673637),
     (249, 0.0018864477853909112),
     (251, 0.00835410836550599),
     (252, 0.0030487254820830752),
     (253, 0.0006466337090765656),
     (261, 0.001326564397575043),
     (269, 0.0006466337090765656),
     (271, 0.0008384212379515161),
     (276, 0.00993401998478309),
     (278, 0.003433906304684499),
     (280, 0.002128718568167805),
     (283, 0.021408177039733592),
     (285, 0.00141914571211187),
     (287, 0.0013317723297408249),
     (288, 0.006894771176602813),
     (290, 0.0035478642802796744),
     (291, 0.0018228783452341711),
     (296, 0.009851449545930315),
     (298, 0.04274758129493744),
     (300, 0.0016768424759030323),
     (301, 0.0006122604911884814),
     (302, 0.00425743713633561),
     (303, 0.004177054182752995),
     (304, 0.0020885270913764977),
     (311, 0.0016326946431692835),
     (313, 0.007095728560559349),
     (323, 0.0047656786344253245),
     (325, 0.021633204021993008),
     (327, 0.0017347380583673637),
     (329, 0.017926405059167313),
     (335, 0.00044392410991360827),
     (336, 0.002395196062681588),
     (338, 0.0020885270913764977),
     (339, 0.00141914571211187),
     (344, 0.00141914571211187),
     (345, 0.011542026857753816),
     (346, 0.00020408683039616044),
     (348, 0.0006288159284636372),
     (349, 0.0006466337090765656),
     (351, 0.0036457566904683422),
     (354, 0.007990633978444949),
     (356, 0.0018864477853909112),
     (358, 0.013789542353205626),
     (359, 0.0022632179817679795),
     (362, 0.001016241827361025),
     (367, 0.002395196062681588),
     (369, 0.19796793077217406),
     (370, 0.0437490802856201),
     (371, 0.000510217075990401),
     (372, 0.002395196062681588),
     (373, 0.0016326946431692835),
     (374, 0.0017347380583673637),
     (375, 0.0019388248887635242),
     (376, 0.006497764594124249),
     (377, 0.0030487254820830752),
     (380, 0.001616584272691414),
     (387, 0.000709572856055935),
     (388, 0.00425743713633561),
     (389, 0.0005723177174474165),
     (391, 0.002395196062681588),
     (400, 0.00203248365472205),
     (406, 0.013789542353205626),
     (407, 0.0005723177174474165),
     (409, 0.0035632902612939437),
     (411, 0.000709572856055935),
     (412, 0.0047573726754963535),
     (418, 0.0027579084706411254),
     (421, 0.0027579084706411254),
     (424, 0.007548300596004301),
     (425, 0.0081299346188882),
     (426, 0.003233168545382828),
     (429, 0.0008878482198272165),
     (431, 0.0008878482198272165),
     (432, 0.01016241827361025),
     (433, 0.0016326946431692835),
     (434, 0.0040649673094441),
     (436, 0.002128718568167805),
     (442, 0.009224447128727154),
     (446, 0.004177054182752995),
     (448, 0.02158934687725154),
     (449, 0.03296002310897991),
     (450, 0.0035478642802796744),
     (453, 0.0011224775671788824),
     (454, 0.0040649673094441),
     (456, 0.011316089908839899),
     (457, 0.0020885270913764977),
     (458, 0.0003233168545382828),
     (463, 0.017926405059167313),
     (464, 0.0027579084706411254),
     (465, 0.000857004029316771),
     (468, 0.00203248365472205),
     (470, 0.0018228783452341711),
     (478, 0.010937270071405025),
     (482, 0.03449095189734856),
     (484, 0.013789542353205626),
     (486, 0.0020885270913764977),
     (489, 0.0054686350357025125),
     (490, 0.022632179817679798),
     (491, 0.0040649673094441),
     (493, 0.000709572856055935),
     (497, 0.012855060439751563),
     (498, 0.0035478642802796744),
     (502, 0.006386155704503415),
     (505, 0.011985950967667422),
     (507, 0.0009699505636148484),
     (514, 0.0025152637138545486),
     (520, 0.00428582343831937),
     (524, 0.0020885270913764977),
     (526, 0.0057231771744741654),
     (527, 0.0027579084706411254),
     (529, 0.006326691742280973),
     (531, 0.003551392879308866),
     (532, 0.012807330322218934),
     (534, 0.0053270893189632995),
     (536, 0.0014286078127731233),
     (541, 0.001144635434894833),
     (543, 0.011119916813659092),
     (544, 0.0031440796423181853),
     (546, 0.0017169531523422495),
     (551, 0.004790392125363176),
     (552, 0.0031771190896168833),
     (558, 0.002395196062681588),
     (559, 0.00020408683039616044),
     (561, 0.002244955134357765),
     (562, 0.005204214175102091),
     (563, 0.006916975213100009),
     (564, 0.001144635434894833),
     (565, 0.0040649673094441),
     (566, 0.0013789542353205627),
     (569, 0.04039709400213835),
     (573, 0.010643592840839025),
     (576, 0.0025152637138545486),
     (578, 0.0028615885872370827),
     (579, 0.0031633458711404867),
     (582, 0.0035478642802796744),
     (586, 0.003568029506622265),
     (590, 0.001714008058633542),
     (592, 0.0010480265474393951),
     (593, 0.0018228783452341711),
     (594, 0.015918772770900515),
     (596, 0.0015885595448084417),
     (598, 0.002395196062681588),
     (600, 0.012531162548258987),
     (601, 0.00203248365472205),
     (604, 0.05589330050485638),
     (605, 0.006386155704503415),
     (606, 0.0005723177174474165),
     (616, 0.0010480265474393951),
     (619, 0.007113692791527175),
     (620, 0.024976206546355857),
     (626, 0.0013789542353205627),
     (628, 0.00041921061897575807),
     (629, 0.004203119108997676),
     (633, 0.0013317723297408249),
     (635, 0.004967009992391545),
     (636, 0.005999028205217397),
     (641, 0.006354238179233767),
     (646, 0.0027579084706411254),
     (649, 0.005771013428876908),
     (652, 0.0009183907367827219),
     (653, 0.001144635434894833),
     (654, 0.0023786863377481767),
     (655, 0.0016768424759030323),
     (657, 0.0030487254820830752),
     (658, 0.007112970799842221),
     (660, 0.005150859457026749),
     (662, 0.003568029506622265),
     (663, 0.0011893431688740884),
     (665, 0.001016241827361025),
     (666, 0.006403665161109467),
     (667, 0.0162598692377764),
     (668, 0.00567658284844748),
     (670, 0.011737897331321226),
     (674, 0.0024490419647539255),
     (675, 0.045065141539893994),
     (676, 0.061732838476866336),
     (678, 0.0004081736607923209),
     (679, 0.005150859457026749),
     (681, 0.0007143039063865616),
     (682, 0.0015885595448084417),
     (685, 0.0020885270913764977),
     (686, 0.0006466337090765656),
     (693, 0.003571519531932808),
     (698, 0.002289270869789666),
     (701, 0.0006122604911884814),
     (705, 0.003428016117267084),
     (712, 0.001714008058633542),
     (713, 0.0013789542353205627),
     (720, 0.0029344743328303066),
     (722, 0.005081209136805125),
     (726, 0.017739321401398375),
     (727, 0.004401711499245459),
     (728, 0.0031771190896168833),
     (729, 0.0060974509641661505),
     (730, 0.010301718914053497),
     (731, 0.0017169531523422495),
     (740, 0.008367560046242579),
     (741, 0.006707369903612129),
     (743, 0.006938952233469455),
     (744, 0.010346139345225049),
     (745, 0.0062655812741294935),
     (748, 0.017276111065137423),
     (750, 0.07136059013244529),
     (752, 0.14617200834752261),
     (753, 0.06989661997157144),
     (758, 0.004203119108997676),
     (769, 0.001714008058633542),
     (770, 0.0010480265474393951),
     (771, 0.0007143039063865616),
     (776, 0.00020960530948787903),
     (783, 0.006354238179233767),
     (784, 0.004177054182752995),
     (785, 0.00283829142422374),
     (786, 0.004177054182752995),
     (787, 0.006938952233469455),
     (788, 0.0029592590407443264),
     (789, 0.003551392879308866),
     (793, 0.014225941599684442),
     (797, 0.004203119108997676),
     (798, 0.024389803856664602),
     (803, 0.01016241827361025),
     (805, 0.003428016117267084),
     (806, 0.02574759837498928),
     (810, 0.0011893431688740884),
     (811, 0.006894771176602813),
     (815, 0.0018864477853909112),
     (816, 0.0012576318569272743),
     (817, 0.0015885595448084417),
     (819, 0.0026635446594816497),
     (820, 0.006466337090765656),
     (821, 0.006214937538790517),
     (822, 0.06898190379469712),
     (825, 0.0003233168545382828),
     (830, 0.0012245209823769628),
     (831, 0.0035564853999211104),
     (832, 0.02259752020860768),
     (833, 0.010992773054301614),
     (834, 0.001714008058633542),
     (838, 0.0019388248887635242),
     (839, 0.0016326946431692835),
     (842, 0.0013789542353205627),
     (848, 0.006894771176602813),
     (849, 0.001144635434894833),
     (851, 0.0020885270913764977),
     (857, 0.0005723177174474165),
     (859, 0.00835410836550599),
     (861, 0.0018864477853909112),
     (862, 0.0017347380583673637),
     (866, 0.004439241099136083),
     (867, 0.004790392125363176),
     (870, 0.0003233168545382828),
     (873, 0.018886484675764745),
     (875, 0.0009699505636148484),
     (878, 0.00203248365472205),
     (879, 0.005496386527150807),
     (881, 0.000709572856055935),
     (883, 0.0008163473215846418),
     (885, 0.002395196062681588),
     (887, 0.0017347380583673637),
     (891, 0.0006466337090765656),
     (892, 0.0013789542353205627),
     (894, 0.0024490419647539255),
     (897, 0.005999028205217397),
     (898, 0.001144635434894833),
     (903, 0.0011893431688740884),
     (905, 0.0025710120879503125),
     (909, 0.002653128795150086),
     (910, 0.005515816941282251),
     (917, 0.05884366752596747),
     (919, 0.001714008058633542),
     (921, 0.0011224775671788824),
     (924, 0.005408301005498252),
     (925, 0.004790392125363176),
     (929, 0.013735625218737996),
     (931, 0.001016241827361025),
     (935, 0.0026635446594816497),
     (936, 0.0025152637138545486),
     (937, 0.0009699505636148484),
     (939, 0.0016326946431692835),
     (944, 0.0008878482198272165),
     (945, 0.007113692791527175),
     (948, 0.0023786863377481767),
     (951, 0.012708476358467533),
     (952, 0.0060974509641661505),
     (953, 0.005081209136805125),
     (956, 0.01016241827361025),
     (957, 0.0012576318569272743),
     (958, 0.000857004029316771),
     (962, 0.001144635434894833),
     (966, 0.12769360036819888),
     (967, 0.047763365885968176),
     (971, 0.0019399011272296968),
     (973, 0.000510217075990401),
     (975, 0.0004081736607923209),
     (976, 0.002040868303961604),
     (980, 0.009514745350992707),
     (981, 0.016547450823846753),
     (982, 0.001016241827361025),
     (985, 0.03891760478642432),
     (988, 0.004387866853517449),
     (992, 0.0014286078127731233),
     (994, 0.007942797724042208),
     (995, 0.021752281385766806),
     (996, 0.011542026857753816),
     (997, 0.0022632179817679795),
     (998, 0.03018316456625458),
     (999, 0.009531357268850649),
     (1000, 0.00010204341519808022),
     (1004, 0.0011893431688740884),
     (1007, 0.01135316569689496),
     (1009, 0.004526435963535959),
     (1010, 0.003433906304684499),
     (1012, 0.013211143755693325),
     (1013, 0.0060974509641661505),
     (1016, 0.01877598839644676),
     (1018, 0.005918518081488653),
     (1019, 0.00020408683039616044),
     (1020, 0.0007143039063865616),
     (1022, 0.002289270869789666),
     (1023, 0.0012932674181531311),
     (1024, 0.056056255628418865),
     (1026, 0.0017169531523422495),
     (1029, 0.007113692791527175),
     (1030, 0.0030487254820830752),
     (1031, 0.0006122604911884814),
     (1037, 0.000709572856055935),
     (1039, 0.0005723177174474165),
     (1042, 0.011119916813659092),
     (1045, 0.020031120110659578),
     (1048, 0.007713036263850938),
     (1050, 0.0015885595448084417),
     (1053, 0.006867812609368998),
     (1060, 0.0024490419647539255),
     (1061, 0.01971109267428573),
     (1062, 0.004285020146583855),
     (1065, 0.009652679647243938),
     (1067, 0.0023786863377481767),
     (1068, 0.009322406308185774),
     (1077, 0.0011893431688740884),
     (1082, 0.004136862705961688),
     (1083, 0.06982276152858481),
     (1085, 0.0162598692377764),
     (1086, 0.025062325096517974),
     (1087, 0.00141914571211187),
     (1088, 0.0005723177174474165),
     (1089, 0.009699505636148483),
     (1092, 0.11904604420846036),
     (1093, 0.011119916813659092),
     (1094, 0.0013789542353205627),
     (1095, 0.0018228783452341711),
     (1096, 0.004578541739579332),
     (1098, 0.0006466337090765656),
     (1103, 0.0031771190896168833),
     (1106, 0.0047656786344253245),
     (1107, 0.0022632179817679795),
     (1108, 0.009605497741664202),
     (1109, 0.0018228783452341711),
     (1110, 0.005496386527150807),
     (1112, 0.001775696439654433),
     (1115, 0.022063267765129003),
     (1117, 0.0022632179817679795),
     (1118, 0.004883165209049691),
     (1119, 0.0036457566904683422),
     (1120, 0.01016241827361025),
     (1122, 0.010704088519866796),
     (1123, 0.0031771190896168833),
     (1124, 0.0003233168545382828),
     (1125, 0.0020960530948787902),
     (1132, 0.0006288159284636372),
     (1135, 0.0031771190896168833),
     (1136, 0.0020885270913764977),
     (1142, 0.0012245209823769628),
     (1145, 0.0006122604911884814),
     (1146, 0.001016241827361025),
     (1148, 0.001326564397575043),
     (1154, 0.0015885595448084417),
     (1158, 0.0020885270913764977),
     (1161, 0.0015885595448084417),
     (1167, 0.005515816941282251),
     (1169, 0.00141914571211187),
     (1173, 0.001775696439654433),
     (1174, 0.0031440796423181853),
     (1175, 0.0009699505636148484),
     (1176, 0.0005723177174474165),
     (1177, 0.002395196062681588),
     (1179, 0.022063267765129003),
     (1180, 0.0008384212379515161),
     (1182, 0.0015885595448084417),
     (1185, 0.00020960530948787903),
     (1186, 0.0027579084706411254),
     (1187, 0.00141914571211187),
     (1192, 0.006916975213100009),
     (1193, 0.007113692791527175),
     (1196, 0.0032018325805547336),
     (1198, 0.027354892884104035),
     (1200, 0.0009699505636148484),
     (1208, 0.003433906304684499),
     (1209, 0.0016326946431692835),
     (1210, 0.0022632179817679795),
     (1212, 0.0040649673094441),
     (1214, 0.0036457566904683422),
     (1218, 0.00020960530948787903),
     (1223, 0.018314166958317328),
     (1224, 0.0062655812741294935),
     (1225, 0.006403665161109467),
     (1227, 0.0013789542353205627),
     (1228, 0.004526435963535959),
     (1229, 0.005240132737196976),
     (1230, 0.002142911719159685),
     (1232, 0.008325402182118618),
     (1234, 0.004177054182752995),
     (1236, 0.030186839474125364),
     (1239, 0.0006288159284636372),
     (1240, 0.0016326946431692835),
     (1241, 0.000510217075990401),
     (1245, 0.004285020146583855),
     (1246, 0.0013789542353205627),
     (1247, 0.0028615885872370827),
     (1248, 0.021408177039733592),
     (1249, 0.000709572856055935),
     (1251, 0.008273725411923376),
     (1254, 0.009114391726170856),
     (1255, 0.12843268694612422),
     (1257, 0.0027579084706411254),
     (1258, 0.004006224022131915),
     (1259, 0.01670821673101198),
     (1260, 0.0016326946431692835),
     (1261, 0.002395196062681588),
     (1264, 0.006497764594124249),
     (1267, 0.0018864477853909112),
     (1270, 0.0054686350357025125),
     (1272, 0.000857004029316771),
     (1273, 0.0016326946431692835),
     (1275, 0.004387866853517449),
     (1278, 0.0016326946431692835),
     (1281, 0.10823022836754204),
     (1283, 0.0057231771744741654),
     (1284, 0.0015885595448084417),
     (1285, 0.002289270869789666),
     (1286, 0.0012576318569272743),
     (1290, 0.00425743713633561),
     (1292, 0.00020408683039616044),
     (1293, 0.011178660100971275),
     (1294, 0.0018228783452341711),
     (1296, 0.0008878482198272165),
     (1297, 0.0033536849518060645),
     (1300, 0.0006466337090765656),
     (1301, 0.005150859457026749),
     (1302, 0.005515816941282251),
     (1305, 0.001016241827361025),
     (1307, 0.006789653945303939),
     (1310, 0.0015306512279712034),
     (1311, 0.005449738046684855),
     (1314, 0.02336794208036037),
     (1315, 0.11353165696894958),
     (1317, 0.006288159284636371),
     (1319, 0.011178660100971275),
     (1320, 0.0027551722103481657),
     (1322, 0.001144635434894833),
     (1323, 0.042031191089976765),
     (1324, 0.010937270071405025),
     (1325, 0.0017347380583673637),
     (1326, 0.00020960530948787903),
     (1327, 0.00020408683039616044),
     (1328, 0.0035632902612939437),
     (1331, 0.0022196205495680415),
     (1333, 0.09557705881371856),
     (1334, 0.000709572856055935),
     (1335, 0.0031771190896168833),
     (1336, 0.0015885595448084417),
     (1337, 0.0012245209823769628),
     (1338, 0.002128718568167805),
     (1339, 0.0005723177174474165),
     (1340, 0.001714008058633542),
     (1342, 0.0703266660013487),
     (1346, 0.013789542353205626),
     (1347, 0.0022632179817679795),
     (1349, 0.005999028205217397),
     (1351, 0.0012576318569272743),
     (1355, 0.0025152637138545486),
     (1356, 0.0006122604911884814),
     (1359, 0.0030487254820830752),
     (1360, 0.0014672371664151533),
     (1363, 0.04317869375450308),
     (1364, 0.013255991036069595),
     (1379, 0.002289270869789666),
     (1380, 0.0006288159284636372),
     (1381, 0.0008878482198272165),
     (1390, 0.00851487427267122),
     (1393, 0.0006466337090765656),
     (1396, 0.001144635434894833),
     (1402, 0.000709572856055935),
     (1406, 0.0020885270913764977),
     (1407, 0.008434558088358557),
     (1410, 0.001144635434894833),
     (1411, 0.003233168545382828),
     (1414, 0.00141914571211187),
     (1416, 0.0023786863377481767),
     (1418, 0.0012245209823769628),
     (1423, 0.0053270893189632995),
     (1425, 0.002289270869789666),
     (1426, 0.0025710120879503125),
     (1428, 0.026829479614448516),
     (1430, 0.023442222000449566),
     (1433, 0.006295494891921581),
     (1439, 0.0027579084706411254),
     (1441, 0.0036457566904683422),
     (1442, 0.002395196062681588),
     (1443, 0.0013789542353205627),
     (1445, 0.03847470569005565),
     (1449, 0.0031771190896168833),
     (1450, 0.0008384212379515161),
     (1451, 0.000857004029316771),
     (1457, 0.005408301005498252),
     (1464, 0.004883165209049691),
     (1465, 0.0017169531523422495),
     (1471, 0.03132732846581063),
     (1476, 0.0032018325805547336),
     (1478, 0.0025710120879503125),
     (1480, 0.0009183907367827219),
     (1482, 0.0062655812741294935),
     (1488, 0.001144635434894833),
     (1489, 0.0051730696726125245),
     (1492, 0.0071855881880447635),
     (1493, 0.0010480265474393951),
     (1497, 0.006894771176602813),
     (1500, 0.000510217075990401),
     (1502, 0.005081209136805125),
     (1503, 0.0027551722103481657),
     (1506, 0.024853116850186357),
     (1520, 0.0027248690233424274),
     (1523, 0.016547450823846753),
     (1524, 0.07308600417376131),
     (1525, 0.004177054182752995),
     (1530, 0.0023786863377481767),
     (1532, 0.0036457566904683422),
     (1533, 0.0023786863377481767),
     (1534, 0.002128718568167805),
     (1535, 0.0008384212379515161),
     (1536, 0.02412547710590179),
     (1540, 0.0013789542353205627),
     (1541, 0.008584765761711247),
     (1542, 0.0014286078127731233),
     (1543, 0.009322406308185774),
     (1544, 0.003233168545382828),
     (1546, 0.009580784250726351),
     (1548, 0.006403665161109467),
     (1552, 0.016122859601296675),
     (1554, 0.017840147533111323),
     (1557, 0.0013789542353205627),
     (1559, 0.0017347380583673637),
     (1561, 0.005918518081488653),
     (1566, 0.004967009992391545),
     (1568, 0.013789542353205626),
     (1572, 0.0015885595448084417),
     (1576, 0.0016326946431692835),
     (1577, 0.00020408683039616044),
     (1578, 0.0006122604911884814),
     (1581, 0.012429875077581034),
     (1583, 0.0047656786344253245),
     (1587, 0.0017169531523422495),
     (1588, 0.0013789542353205627),
     (1589, 0.004177054182752995),
     (1590, 0.0020885270913764977),
     (1598, 0.0004081736607923209),
     (1601, 0.0022196205495680415),
     (1605, 0.0023786863377481767),
     (1607, 0.003433906304684499),
     (1609, 0.004790392125363176),
     (1613, 0.0057231771744741654),
     (1616, 0.008273725411923376),
     (1619, 0.000857004029316771),
     (1624, 0.005142024175900625),
     (1625, 0.017276111065137423),
     (1627, 0.001144635434894833),
     (1628, 0.04063455793876657),
     (1629, 0.0024490419647539255),
     (1635, 0.007713036263850938),
     (1636, 0.003568029506622265),
     (1637, 0.013789542353205626),
     (1640, 0.009114391726170856),
     (1642, 0.0035478642802796744),
     (1643, 0.003433906304684499),
     (1644, 0.0020885270913764977),
     (1646, 0.0062655812741294935),
     (1647, 0.0014286078127731233),
     (1648, 0.005999028205217397),
     (1649, 0.000510217075990401),
     (1650, 0.0012576318569272743),
     (1655, 0.003265389286338567),
     (1657, 0.010000254689411863),
     (1661, 0.0016768424759030323),
     (1665, 0.0027551722103481657),
     (1666, 0.0015306512279712034),
     (1667, 0.0012932674181531311),
     (1668, 0.0047573726754963535),
     (1670, 0.03720065163408207),
     (1677, 0.0015306512279712034),
     (1680, 0.001020434151980802),
     (1684, 0.008273725411923376),
     (1686, 0.0014672371664151533),
     (1692, 0.0008163473215846418),
     (1695, 0.0012932674181531311),
     (1696, 0.012194901928332301),
     (1701, 0.005142024175900625),
     (1705, 0.0026635446594816497),
     (1708, 0.0035632902612939437),
     (1710, 0.0012245209823769628),
     (1712, 0.0003061302455942407),
     (1715, 0.0011893431688740884),
     (1716, 0.0017347380583673637),
     (1719, 0.0003233168545382828),
     (1722, 0.0027579084706411254),
     (1727, 0.001144635434894833),
     (1728, 0.10585154202979385),
     (1735, 0.00203248365472205),
     (1743, 0.0020960530948787902),
     (1744, 0.0009699505636148484),
     (1752, 0.026635446594816495),
     (1754, 0.0072915133809366844),
     (1759, 0.008273725411923376),
     (1761, 0.0031440796423181853),
     (1762, 0.0008384212379515161),
     (1763, 0.005946715844370442),
     (1766, 0.0014672371664151533),
     (1768, 0.0071855881880447635),
     (1770, 0.0023786863377481767),
     (1772, 0.016283076557018647),
     (1774, 0.0031771190896168833),
     (1775, 0.0054686350357025125),
     (1778, 0.006867812609368998),
     (1779, 0.0072915133809366844),
     (1780, 0.0010480265474393951),
     (1781, 0.00283829142422374),
     (1782, 0.004967009992391545),
     (1784, 0.0027579084706411254),
     (1785, 0.006403665161109467),
     (1793, 0.0020885270913764977),
     (1794, 0.000857004029316771),
     (1797, 0.01885408864496896),
     (1798, 0.021122986946002606),
     (1799, 0.0047656786344253245),
     (1806, 0.0029592590407443264),
     (1808, 0.006354238179233767),
     (1813, 0.00044392410991360827),
     (1815, 0.0013789542353205627),
     (1816, 0.0013789542353205627),
     (1820, 0.0023786863377481767),
     (1823, 0.00857004029316771),
     (1825, 0.0054686350357025125),
     (1832, 0.0011224775671788824),
     (1833, 0.0031074687693952584),
     (1834, 0.0008384212379515161),
     (1835, 0.000709572856055935),
     (1838, 0.0005723177174474165),
     (1840, 0.0006288159284636372),
     (1841, 0.0015885595448084417),
     (1846, 0.006894771176602813),
     (1847, 0.005150859457026749),
     (1848, 0.009652679647243938),
     (1849, 0.001020434151980802),
     (1850, 0.005515816941282251),
     (1851, 0.00942704432248448),
     (1853, 0.01714008058633542),
     (1854, 0.001714008058633542),
     (1857, 0.004387866853517449),
     (1858, 0.0029098516908445454),
     (1859, 0.04183780023121289),
     (1860, 0.0035564853999211104),
     (1862, 0.007113692791527175),
     (1864, 0.002128718568167805),
     (1866, 0.004136862705961688),
     (1869, 0.006354238179233767),
     (1876, 0.00020960530948787903),
     (1878, 0.0737023465212423),
     (1881, 0.020051661797575882),
     (1884, 0.0060974509641661505),
     (1885, 0.008584765761711247),
     (1898, 0.005819703381689091),
     (1899, 0.0027579084706411254),
     (1904, 0.005515816941282251),
     (1905, 0.005142024175900625),
     (1906, 0.000857004029316771),
     (1907, 0.0006466337090765656),
     (1908, 0.0004081736607923209),
     (1909, 0.003428016117267084),
     (1910, 0.0027579084706411254),
     (1915, 0.002395196062681588),
     (1916, 0.0013317723297408249),
     (1922, 0.012531162548258987),
     (1923, 0.0031771190896168833),
     (1926, 0.0025710120879503125),
     (1933, 0.0018367814735654438),
     (1934, 0.0003061302455942407),
     (1935, 0.007713036263850938),
     (1938, 0.006214937538790517),
     (1940, 0.006856032234534168),
     (1941, 0.0033536849518060645),
     (1942, 0.00010204341519808022),
     (1944, 0.0011893431688740884),
     (1945, 0.0012576318569272743),
     (1948, 0.0005723177174474165),
     (1949, 0.005000127344705931),
     (1951, 0.010704088519866796),
     (1952, 0.0023786863377481767),
     (1953, 0.0027579084706411254),
     (1958, 0.004790392125363176),
     (1959, 0.00020960530948787903),
     (1964, 0.0072915133809366844),
     (1965, 0.001016241827361025),
     (1966, 0.010992773054301614),
     (1967, 0.005000127344705931),
     (1968, 0.005142024175900625),
     (1974, 0.014191457121118698),
     (1979, 0.002128718568167805),
     (1980, 0.004285020146583855),
     (1981, 0.002040868303961604),
     (1982, 0.0013789542353205627),
     (1985, 0.015885595448084416),
     (1986, 0.061704290110807504),
     (1990, 0.01197598031340794),
     (1991, 0.0003233168545382828),
     (1993, 0.02068431352980844),
     (1994, 0.009146176446249226),
     (1997, 0.0025865348363062622),
     (2000, 0.0025152637138545486),
     (2001, 0.0007143039063865616),
     (2002, 0.0017347380583673637),
     (2003, 0.001016241827361025),
     (2007, 0.0016326946431692835),
     (2010, 0.003982500880269702),
     (2012, 0.013624345116712136),
     (2013, 0.0013317723297408249),
     (2014, 0.0011224775671788824),
     (2018, 0.00141914571211187),
     (2020, 0.0016326946431692835),
     (2021, 0.001714008058633542),
     (2022, 0.008265516631044498),
     (2023, 0.0005723177174474165),
     (2026, 0.004820922118221218),
     (2030, 0.0005723177174474165),
     (2031, 0.0018228783452341711),
     (2032, 0.0020885270913764977),
     (2037, 0.0005723177174474165),
     (2039, 0.013255991036069595),
     (2044, 0.005946715844370442),
     (2045, 0.0010480265474393951),
     (2049, 0.0035478642802796744),
     (2051, 0.0008163473215846418),
     (2053, 0.003568029506622265),
     (2054, 0.0008878482198272165),
     (2055, 0.0017347380583673637),
     (2065, 0.0018367814735654438),
     (2066, 0.0005723177174474165),
     (2067, 0.00993401998478309),
     (2068, 0.0020885270913764977),
     (2069, 0.0005723177174474165),
     (2073, 0.0047573726754963535),
     (2074, 0.001016241827361025),
     (2076, 0.0031074687693952584),
     (2078, 0.0015885595448084417),
     (2082, 0.0011893431688740884),
     (2083, 0.0014672371664151533),
     (2084, 0.001144635434894833),
     (2086, 0.0054686350357025125),
     (2087, 0.011318686712345467),
     (2088, 0.0003233168545382828),
     (2089, 0.0009183907367827219),
     (2090, 0.0032018325805547336),
     (2095, 0.0013789542353205627),
     (2096, 0.0008878482198272165),
     (2102, 0.0027579084706411254),
     (2108, 0.004177054182752995),
     (2110, 0.000857004029316771),
     (2111, 0.0009183907367827219),
     (2114, 0.007095728560559349),
     (2116, 0.0023786863377481767),
     (2117, 0.004136862705961688),
     (2118, 0.02068431352980844),
     (2119, 0.004849752818074242),
     (2125, 0.0011893431688740884),
     (2127, 0.002395196062681588),
     (2128, 0.005946715844370442),
     (2133, 0.00141914571211187),
     (2134, 0.0038798022544593936),
     (2135, 0.0020885270913764977),
     (2136, 0.007942797724042208),
     (2138, 0.002128718568167805),
     (2144, 0.039444656253670496),
     (2145, 0.00141914571211187),
     (2148, 0.002244955134357765),
     (2152, 0.0023056584043666694),
     (2154, 0.0014286078127731233),
     (2155, 0.0054686350357025125),
     (2156, 0.04136862705961688),
     (2158, 0.07768671923488145),
     (2159, 0.024389803856664602),
     (2162, 0.014307942936185413),
     (2164, 0.32782265949784195),
     (2165, 0.007546709868531341),
     (2169, 0.0025510853799520054),
     (2170, 0.0012932674181531311),
     (2172, 0.00203248365472205),
     (2176, 0.0060974509641661505),
     (2180, 0.0029344743328303066),
     (2183, 0.0026635446594816497),
     (2186, 0.028615885872370826),
     (2187, 0.045421213553788864),
     (2195, 0.013735625218737996),
     (2197, 0.02841114303447093),
     (2200, 0.000709572856055935),
     (2202, 0.0013789542353205627),
     (2206, 0.016597213805975078),
     (2208, 0.001016241827361025),
     (2210, 0.02620013047109069),
     (2222, 0.0018228783452341711),
     (2223, 0.0018228783452341711),
     (2226, 0.002142911719159685),
     (2227, 0.00203248365472205),
     (2229, 0.0054686350357025125),
     (2232, 0.00010204341519808022),
     (2233, 0.0024490419647539255),
     (2234, 0.006354238179233767),
     (2235, 0.0047656786344253245),
     (2237, 0.00041921061897575807),
     (2238, 0.00020408683039616044),
     (2240, 0.0012932674181531311),
     (2241, 0.009652679647243938),
     (2242, 0.00283829142422374),
     (2244, 0.07446352870731038),
     (2249, 0.0008878482198272165),
     (2250, 0.0022196205495680415),
     (2255, 0.0057231771744741654),
     (2258, 0.002395196062681588),
     (2264, 0.0003061302455942407),
     (2266, 0.0020885270913764977),
     (2267, 0.004611316808733339),
     (2272, 0.004136862705961688),
     (2273, 0.004177054182752995),
     (2274, 0.0027551722103481657),
     (2277, 0.002128718568167805),
     (2279, 0.002395196062681588),
     (2280, 0.0051730696726125245),
     (2281, 0.0008384212379515161),
     (2282, 0.0017347380583673637),
     (2284, 0.013082774857614972),
     (2285, 0.0023786863377481767),
     (2289, 0.006428735157479055),
     (2290, 0.005306257590300172),
     (2292, 0.004285020146583855),
     (2294, 0.0015885595448084417),
     (2296, 0.0015885595448084417),
     (2297, 0.00041921061897575807),
     (2300, 0.004006224022131915),
     (2302, 0.009114391726170856),
     (2303, 0.002040868303961604),
     (2305, 0.005515816941282251),
     (2309, 0.01422738558305435),
     (2311, 0.008273725411923376),
     (2313, 0.007440130326816415),
     (2315, 0.0008878482198272165),
     (2317, 0.002244955134357765),
     (2319, 0.002289270869789666),
     (2320, 0.0031074687693952584),
     (2322, 0.003995316989222474),
     (2325, 0.14837130679470964),
     (2328, 0.001016241827361025),
     (2330, 0.04292382880855624),
     (2332, 0.0020960530948787902),
     (2335, 0.004006224022131915),
     (2336, 0.011178660100971275),
     (2337, 0.0031771190896168833),
     (2339, 0.0030613024559424068),
     (2340, 0.03723176435365519),
     (2342, 0.0014672371664151533),
     (2343, 0.0025710120879503125),
     (2344, 0.0014672371664151533),
     (2346, 0.004967009992391545),
     (2349, 0.003428016117267084),
     (2352, 0.0013317723297408249),
     (2353, 0.0009699505636148484),
     (2357, 0.0688285670374257),
     (2359, 0.0015885595448084417),
     (2361, 0.002653128795150086),
     (2363, 0.0006466337090765656),
     (2364, 0.0074491693094598565),
     (2368, 0.027438529338747675),
     (2369, 0.00425743713633561),
     (2370, 0.0013789542353205627),
     (2375, 0.007990633978444949),
     (2376, 0.0031633458711404867),
     (2377, 0.0030613024559424068),
     (2378, 0.008584765761711247),
     (2381, 0.00203248365472205),
     (2382, 0.0016768424759030323),
     (2383, 0.0008878482198272165),
     (2385, 0.0006466337090765656),
     (2387, 0.0003233168545382828),
     (2389, 0.00425743713633561),
     (2392, 0.0006288159284636372),
     (2393, 0.002040868303961604),
     (2396, 0.0062655812741294935),
     (2399, 0.011098102747840208),
     (2401, 0.001016241827361025),
     (2408, 0.016558819449542444),
     (2409, 0.019029490701985414),
     (2417, 0.000709572856055935),
     (2418, 0.004136862705961688),
     (2419, 0.0013789542353205627),
     (2420, 0.0013317723297408249),
     (2421, 0.010654178637926599),
     (2422, 0.014205571517235465),
     (2423, 0.0019399011272296968),
     (2431, 0.0005723177174474165),
     (2432, 0.0031633458711404867),
     (2433, 0.00010204341519808022),
     (2434, 0.0020960530948787902),
     (2444, 0.0005723177174474165),
     ...]



## 9. The results of the tf-idf model
<p>Once again, the format of those results is hard to interpret for a human. Therefore, we will transform it into a more readable version and display the 10 most specific words for the "<em>On the Origin of Species</em>" book.</p>


```python
# Convert the tf-idf model for "On the Origin of Species" into a DataFrame
df_tfidf = pd.DataFrame(model[bows[ori]])

# Name the columns of the DataFrame id and score
df_tfidf.columns=["id", "score"]

# Add the tokens corresponding to the numerical indices for better readability
df_tfidf['token'] = [dictionary[i] for i in list(df_tfidf["id"])]

# Sort the DataFrame by descending tf-idf score and print the first 10 rows.
df_tfidf.sort_values(by="score", ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>score</th>
      <th>token</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>880</th>
      <td>2164</td>
      <td>0.327823</td>
      <td>select</td>
    </tr>
    <tr>
      <th>3103</th>
      <td>10108</td>
      <td>0.204162</td>
      <td>pigeon</td>
    </tr>
    <tr>
      <th>128</th>
      <td>369</td>
      <td>0.197968</td>
      <td>breed</td>
    </tr>
    <tr>
      <th>2985</th>
      <td>9395</td>
      <td>0.167705</td>
      <td>migrat</td>
    </tr>
    <tr>
      <th>947</th>
      <td>2325</td>
      <td>0.148371</td>
      <td>steril</td>
    </tr>
    <tr>
      <th>285</th>
      <td>752</td>
      <td>0.146172</td>
      <td>domest</td>
    </tr>
    <tr>
      <th>504</th>
      <td>1255</td>
      <td>0.128433</td>
      <td>hybrid</td>
    </tr>
    <tr>
      <th>371</th>
      <td>966</td>
      <td>0.127694</td>
      <td>fertil</td>
    </tr>
    <tr>
      <th>3840</th>
      <td>16046</td>
      <td>0.124547</td>
      <td>gärtner</td>
    </tr>
    <tr>
      <th>3536</th>
      <td>12729</td>
      <td>0.121348</td>
      <td>naturalis</td>
    </tr>
  </tbody>
</table>
</div>



## 10. Compute distance between texts
<p>The results of the tf-idf algorithm now return stemmed tokens which are specific to each book. We can, for example, see that topics such as selection, breeding or domestication are defining "<em>On the Origin of Species</em>" (and yes, in this book, Charles Darwin talks quite a lot about pigeons too). Now that we have a model associating tokens to how specific they are to each book, we can measure how related to books are between each other.</p>
<p>To this purpose, we will use a measure of similarity called <strong>cosine similarity</strong> and we will visualize the results as a distance matrix, i.e., a matrix showing all pairwise distances between Darwin's books.</p>


```python
# Load the library allowing similarity computations
from gensim import similarities

# Compute the similarity matrix (pairwise distance between all texts)
sims = similarities.MatrixSimilarity(model[bows])

# Transform the resulting list into a DataFrame
sim_df = pd.DataFrame(list(sims))

# Add the titles of the books as columns and index of the DataFrame
sim_df.columns = titles
sim_df.index = titles

# Print the resulting matrix
sim_df
```

## 11. The book most similar to "On the Origin of Species"
<p>We now have a matrix containing all the similarity measures between any pair of books from Charles Darwin! We can now use this matrix to quickly extract the information we need, i.e., the distance between one book and one or several others. </p>
<p>As a first step, we will display which books are the most similar to "<em>On the Origin of Species</em>," more specifically we will produce a bar chart showing all books ranked by how similar they are to Darwin's landmark work.</p>


```python
# This is needed to display plots in a notebook
%matplotlib inline

# Import the needed functions from matplotlib
import matplotlib.pyplot as plt

# Select the column corresponding to "On the Origin of Species" and 
v = sim_df["OriginofSpecies"]

# Sort by ascending scores
v_sorted = v.sort_values(ascending=True)

# Plot this data has a horizontal bar plot
v_sorted.plot.barh(x='lab', y='val', rot=0).plot()

# Modify the axes labels and plot title for better readability
plt.xlabel("Cosine distance")
plt.ylabel("")
plt.title("Most similar books to 'On the Origin of Species'")
```

## 12. Which books have similar content?
<p>This turns out to be extremely useful if we want to determine a given book's most similar work. For example, we have just seen that if you enjoyed "<em>On the Origin of Species</em>," you can read books discussing similar concepts such as "<em>The Variation of Animals and Plants under Domestication</em>" or "<em>The Descent of Man, and Selection in Relation to Sex</em>." If you are familiar with Darwin's work, these suggestions will likely seem natural to you. Indeed, <em>On the Origin of Species</em> has a whole chapter about domestication and <em>The Descent of Man, and Selection in Relation to Sex</em> applies the theory of natural selection to human evolution. Hence, the results make sense.</p>
<p>However, we now want to have a better understanding of the big picture and see how Darwin's books are generally related to each other (in terms of topics discussed). To this purpose, we will represent the whole similarity matrix as a dendrogram, which is a standard tool to display such data. <strong>This last approach will display all the information about book similarities at once.</strong> For example, we can find a book's closest relative but, also, we can visualize which groups of books have similar topics (e.g., the cluster about Charles Darwin personal life with his autobiography and letters). If you are familiar with Darwin's bibliography, the results should not surprise you too much, which indicates the method gives good results. Otherwise, next time you read one of the author's book, you will know which other books to read next in order to learn more about the topics it addressed.</p>


```python
# Import libraries
from scipy.cluster import hierarchy

# Compute the clusters from the similarity matrix,
# using the Ward variance minimization algorithm
Z = hierarchy.linkage(sim_df, 'ward')

# Display this result as a horizontal dendrogram
a = hierarchy.dendrogram(Z,  leaf_font_size=8, labels=sim_df.index,  orientation="left")
```
