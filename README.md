# Attention (please!)
## Sequence classification Analysis

Update your B.Sc. / M.Sc. thesis title classification model from last time.
In case you want to start fresh, we provide some boiler plate code of a base model as well
as ready-to-go data loading. 
If you want to dive deeper into how padding and loss calculation of sequences of different lengths works in pytorch
you can check out the short tutorial code file that can be found under ./res  along the base model implementation.

### Instructions
* Implement **dot product attention** and check how it affects the training.
* Do your results improve, compared to your old model or the base model?
* Can you find certain words that have high attention weights regarding the decision?

## Sentiment Analysis
We will use the kaggle Rotten Tomates dataset for this exercise.
[Source and Download instructions](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data)
The dataset is comprised of tab-separated files with phrases from the Rotten Tomatoes dataset. The train/test split has been preserved for the purposes of benchmarking, but the sentences have been shuffled from their original order. Each Sentence has been parsed into many phrases by the Stanford parser. Each phrase has a PhraseId. Each sentence has a SentenceId. Phrases that are repeated (such as short/common words) are only included once in the data.

train.tsv contains the phrases and their associated sentiment labels. We have additionally provided a SentenceId so that you can track which phrases belong to a single sentence.
test.tsv contains just phrases. You must assign a sentiment label to each phrase.

The sentiment labels are:

* 0 - negative
* 1 - somewhat negative
* 2 - neutral
* 3 - somewhat positive
* 4 - positive

### Links
* https://torchtext.readthedocs.io/en/latest/vocab.html
* https://www.cs.toronto.edu/~lczhang/360/lec/w06/w2v.html
* https://nlp.stanford.edu/courses/cs224n/2012/reports/WuJean_PaoYuanyuan_224nReport.pdf

### Instructions
Use GloVe word embeddings, there is a number of pretrained models for English
available in the torchtext module.
You are free to use any kind of attention and architecture you like.
Just remember that the basic form for attention based networks is always an
encoder / decoder architecture.

To get you started quickly with the word embeddings use torchtext and download
the English challenge data from kaggle.
```python
import torch
import torchtext

# The first time you run this will download a ~823MB file
glove = torchtext.vocab.GloVe(name="6B", # trained on Wikipedia 2014 corpus
                              dim=50)   # embedding size = 100
```
* Check your classification results. Can you beat the baseline (there are plenty of implemntations/baselines you can find on the internet)?
* What words influence the decision most?
* Visualize the attention weights for the words and pick some nice samples!

<!-- https://deeplearning.cs.cmu.edu/F20/document/homework/Homework_4_2.pdf -->
