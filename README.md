# Project 3

## Project Overview
### 1. Problem Statement

This project at its core, is about understanding if a model is capable of detecting unethical text. On the internet, as in life,  
people often behave in a variety of ways with regards to ethicality. This project will attempt to detect unethical text using  
Reddit subreddits and aims to help understand the fundamental characteristics of unethical text so that it may be identified in  
other forums.

### 2. Executive Summary
The project begins with the selection of two subreddits, or forums/communities within reddit.com, that are purportedly ethical  
opposites. The subreddit 'LifeProTips' claims to be a community where only ethical life protips, or advice, are presented. The   
rules disallow unethical protips, and the title of the post should give the gist of the protip. The subreddit 'UnethicalLifeProTips'  
claims to be a community where protips must be unethical, and the title must explain the protip without needing a description.  
By selecting these two communities, I have access to data in the same general space (life protips) that can easily be labeled  
'ethical' or 'unethical' depending on the subreddit that it comes from. My assumption when analyzing these subreddits are that  
the moderators of each subreddit are active and doing their due diligence to remove posts that do not comply with the  
subreddit rules. 

Once the subreddits are selected, a number of posts are scraped using the Reddit API. The number of posts scraped are enough to   
provide an ample number of samples with which to train and test the classifier. The minimum recommended posts was 1000, however   
by automating the process 20,000 raw posts were obtained. The posts were then trimmed to remove any that had NaN values for post  
or text, in case either data field was to be used in the model. Identifying words and phrases such as LPT and ULPT and all of their  
variations were removed by regex, so that the classifier did not have an easy time of identifying the text and instead would rely  
on the content of the title and text. The cleaned data was analyzed using term frequency, post length, subjectivity and polarity values,  
and sentiment analysis to determine if any key attributes of either text or title could give clues as to how to separate the data.  

Following the analysis of the subreddits, models are constructed to see if the two subreddits could be identified by an algorithm.   
Each subreddit is encoded as a zero or one, and in most of the models the words in the text and title of the posts are vectorized using  
a CountVectorizer. Ethical LifeProTips are coded as a zero, since this is our base or reference value, and UnethicalLifeProTips are   
encoded as a 1. Once this is complete, different classification estimators are fit and trained on the data, and then make  
predictions and are scored according to their accuracy. Accuracy is the chosen metric because we want to be able to positively identify  
unethical text. Models are grid searched and cross validated using various parameters to optimize on accurately predicting unethical  
text. Some of these parameters include the number of important features, Naive Bayes alpha score, and the LogisticRegression   
regularization parameter, C.  

Overall, the project was somewhat of a success, within the scope of Reddit posts. At the time of this writing the project and models  
have not been generalized to accept all text, but this is a feature to be implemented in the near future. The hope is that by using  
the coefficient matrix and mapping other texts onto the weights that the models have found, a prediction (with a certain % of error)  
can be assigned to a text straight away. This will have value for moderators of software platforms, communities and for analysts of   
news and other forms of media that have an influence on the population that consumes them: Detecting unethical text or unethical  
intent can be used to help prevent negative experiences or immoral ideas from taking root in a community.  
The analysis was able to show that the sentiment and polarity of posts shifted depending on if the posts were ethical or not, and   
the classifiers were able to predict with ~83% accuracy whether or not the post belonged to the unethical subreddit. This is an  
improvement over the ~51% accuracy of the null model. It appears this modeling and analysis technique can be used to detect unethical  
text. It is recommended that when trying to infer the ethicality of a text, that the sentiment and polarity scores be examined as  
they are a good clue as to which way the text leans.  

### 3. File Directory

File directory is as below. I have collapsed the folders other than code and data for brevity.

**/project-3**  
**+-- /code/**   
|   +-- 00_Import_Scraping.ipynb    |  **### Imports the data from Reddit**   
|   +-- 01_Cleaning_Initial.ipynb   |  **### Cleaning up posts**   
|   +-- 02_Cleaning_Final.ipynb     |  **### Final Cleaning and prep for Analysis**  
|   +-- 03_EDA.ipynb                |  **### Analysis and Visualizations**  
|   +-- 04_Modeling.ipynb           |  **### Notebook for Classification Modeling**  
**+-- /data/**  
|   +-- both_cleaned.csv  
|   +-- full_raw_scores.csv  
|   +-- lpt_cleaned.csv   
|   +-- lpt_df_2.csv   
|   +-- lpt_df.csv   
|   +-- lpt_trimmed.csv   
|   +-- ulpt_cleaned.csv   
|   +-- ulpt_df_2.csv   
|   +-- ulpt_df.csv   
|   +-- ulpt_trimmed.csv    
+-- **/images/**  | **###Images output from code.**   
+-- **/presentation/** | **### Presentation files and figures, and .pdf of slides.**  
+-- **/scratch/** | **### Scratch folder for work. Currently contains vader work.**  
+-- **/project_readme.md** | **### This file.**  

### 4. Data/Data Dictionary  


The data for this project was scraped from reddit.com, using the reddit api located at this link:  

|                   Reddit API URL                  |
|:-------------------------------------------------:|
| https://api.pushshift.io/reddit/search/submission |

Using the api guide and parameters for pulling posts, a function to pull ('n' * 100) number of posts  
was created. This automated the download process, and used a try except block to avoid errors with the   
encoding as the posts were pulled in. Data was trimmed to remove blank entries and entries that had  
been removed from reddit due to one cause or another.

The finalized dataset had identifying phrases removed from each post text and title, and is stored as  
'both_cleaned.csv'. A second dataset with additional features such as polarity and sentiment is saved  
as 'full_raw_scores.csv'. A breakdown of the columns in this dataframe are as below:  



| Column | both_cleaned.csv | full_raw_scores.csv | Data Type | Notes |
|:-:|:-:|:-:|-|-|
| Unnamed: 0 | X | X | int | artifact of a bad import. Should be removed. |
| title | X | X | str | Reddit Post title. |
| selftext | X | X | str | Reddit post text. |
| title_word_count |  | X | int | Number of words in the title. |
| text_word_count |  | X | int | Number of words in the post text. |
| polarity |  | X | float | TextBlob Polarity score of the title. |
| subjectivity |  | X | float | TextBlob Subjectivity score of the title. |
| title_scores |  | X | dict | SentimentAnalyzer sentiment title scores. Should be ignored during modeling - use other  columns. |
| text_scores |  | X | dict | SentimentAnalyzer sentiment text scores. Should be ignored during modeling - use other columns. |
| title_neg |  | X | float | Title negative sentiment score. |
| title_neu |  | X | float | Title neutral sentiment score. |
| title_pos |  | X | float | Title positive sentiment score. |
| title_comp |  | X | float | Title composite sentiment score. |
| text_neg |  | X | float | Text negative sentiment score. |
| text_neu |  | X | float | Text neutral sentiment score. |
| text_pos |  | X | float | Text positive sentiment score. |
| text_comp |  | X | float | Text composite sentiment score. |
| subreddit | X |  | str | Target column. Typically encoded before modeling. |
| target |  | X | int | Target column, 0 = ethical, 1 = unethical. |

Final datasets were transformed to be used in the predictor models. The vectorizer in each case takes the total number  
of occurrences of that word for that entry, and stores it in a sparse matrix.

### 5. Conclusions and Recommendations

In conclusion, it appears that unethical text can be detected, with reasonable accuracy. Ethical text exhibits more  
positive sentiment and polarity, while unethical text has negative sentiment and less positive polarity. The
verbosity of posts can also indicate if the text is more likely to be ethical, with longer posts more likely to
exhibit ethical characteristics. 
Also more importantly, on the generalization of the model, it has become apparent from analysis of the coefficients
and key words that the model uses, that the unethical tips it has trained on are mostly ideas from the recent past-
This I believe is a result of the dataset that was used to train the model. Future models should include even more
data both to increase the vocabulary and decrease the sensitivity over time to current events influencing what is
ethical or unethical. Of note is the appearance of 'coronavirus' as one of the words with the highest unethical
coefficient - a word that until 2020 would have almost no bearing on whether or not an act was ethical except
perhaps in a medical setting. Future models should take into account historical weights of these variables, to
prevent current events from determining all that is ethical or unethical. Alternatively, this program could be
utilized as a sort of 'barometer' to measure the change in ethics issues around current events. For example, as the
weight for the word 'coronavirus' increased over time it would become clear that ethics and ethical issues around
that idea or word were increasing. It could also be used as a sort of crude scam or spam indicator, especially when
combined with the 'needing', 'wanting' and other value words that are popular in unethical text. The model appears
to understand that nothing good comes cheap or easy. One thing to note however, the generalized model does not like
the golden rule.  

### 6. Areas for Further Research

I would like to expand the model's ability to detect sentiment and potentially map it with personality
types, or assumed personality types as someone is posting on the internet. I have found a Meyer's Briggs crossover
for text but I would like to implement this in the model to see if the 'personality' behind the post can be
detected or used in differentiating ethical and unethical text.  
I would also like to somehow use sarcasm detection to strengthen the model, as sarcasm is a double edged sword
especially on the internet. Sarcasm with respect to ethics can be misconstrued, and/or depending on the reader's
sense of humor, unethical text may be entertaining. Clearly there is a market for such humor or at least interest
in the darker side of things as proven by the existence of Unethical Life Pro Tips, Illegal Life Pro Tips, Sh*!@y
Life Pro Tips, and others.
I think the hardest part about generalizing the model will be finding enough examples of truly unethical text,
of varying lengths, so that the model is more tuned to different types of text. Ultimately, the ethicality of each
of these subreddits depends on the enforcement of the ethical/unethical rules by the moderators. Ethics is also
something that can be deeply personal, and has been debated for millenia. One person's model of ethical behavior
or speech may vary from that of another.
I do also see use in just using these subreddits: Unethical Life Pro Tips essentially acts as a generator
for synthetic data in the underrepresented class. Perhaps my solution is just to sample the ULPT community a
greater amount or over a greater spread of time. The subreddits do have one additional problem, and that is that
tips cannot be repeated. In the real world, unethical ideas can and do get repeated over and over. So the other
flaw of the current model is that since the more recent posts are not allowed to be repeats or simple ideas, the
'base level' or easy ethical and unethical tips may not be captured. Older tips or obvious tips may have been
removed for one reason or another, or users leaving Reddit. This may result in our model missing out on
certain key words or strengths of words that are common in easy to understand, simple life pro tips, rather than
the pro tips being entered today which necessarily have to be different and therefore may be more convoluted.

### 7. Sources:
Building a Logistic Regression Function from Scratch: (also GA labs)
studio-pubs-static.s3.amazonaws.com/74431_8cbd662559f6451f9cd411545f28107f.html
Reference for a Code of Ethics, tested in Generalized Model:
https://www.asha.org/code-of-ethics/  
Twitter snippets from a former president, for testing in Generalized Model: (And understanding the impact of certain  
types of speech on the internet)  
https://www.inc.com/larry-kim/10-unusual-twitter-marketing-tips-from-donald-j-trump.html  
The power of twitter and digital voice:  
https://www.brookings.edu/techstream/how-trump-impacts-harmful-twitter-speech-a-case-study-in-three-tweets/  
Aristotle on excellence and character:  
https://www.forbes.com/quotes/659/  
On Lawyers, and subjectivity of advice:  
https://law.stackexchange.com/questions/4801/why-dont-attorneys-like-to-offer-subjective-advice/4813  

