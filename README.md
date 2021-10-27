# health-fact

## Overview

The goal of this project was to use one of Hugging Face's pretrained transformer models to classify their health_fact dataset. This dataset has a few features which correspond
to nontrivial amounts of natural language, among them the main_text, claim, and explanation. I took an approach kind of similar to question answering with regard to the text
I used, which was to essentially just take the claim and concatenate it with the explanation. I thought this would be the most prudent initial path to take because it could
capture the most meaning which relates to the label while being succint. The main_text feature especially was extremely long, and so I didn't think it was a good idea to use it
because I would probably be forced to truncate the vast majority of it anyway.

I used the base DistilBERT as my transformer of choice. For me it was the natural choice because I have the most experience using it, and it does have the nice property of
being a relatively more lightweight transformer. After running the model inputs through DistilBERT, I retrieved the last hidden state for the leading [CLS] token and fed it
through to a linear layer to get the logits for all 4 potential labels. I thought this would be a relatively simple yet effective approach.

## Preprocessing

Luckily, this dataset was pretty easy to preprocess for the most part. There were only a handful of "unclean" observations that were actually labeled -1, which doesn't 
correspond to any category. I took a look at a couple of these observations and they were not really coherent at all in the context of the rest of the dataset, so I decided to 
just drop them during both training and evaluation. I set up a pipeline that would fetch the relevant dataset split, remove the bad observations as described, and tokenize the 
concatenated claim and explanation (padding and truncating if necessary). This pipeline would yield the necessary input IDS, attention masks, and labels for both training and
validation.

## Training

Training was relatively simple and standard as well. I used stochastic gradient descent as my optimization algorithm and cross-entropy loss as my loss function. What might seem
a little weird though was that I only used a batch size of 2. This is really small, but the reason I did this was so that I could run the training loop locally on my computer
without running out of memory on my GPU. Of course, I could have just use Google Colab or something to get more GPU memory, but I find it more fun to host all the computation
locally for small projects.

## Results

First I trained the model for one epoch. This led to a validation set accuracy of 0.6523887973640856. The confusion matrix that I obtained is listed below.

|   | 0    | 1 | 2    | 3 |
|---|------|---|------|---|
| 0 | 2422 | 0 | 559  | 0 |
| 1 | 1155 | 0 | 279  | 0 |
| 2 | 1335 | 0 | 3743 | 0 |
| 3 | 204  | 0 | 87   | 0 |

And here is a legend to say what the numbered outcomes mean.

| Number | Meaning |
|--------|---------|
| 0      | false   |
| 1      | mixture |
| 2      | true    |
| 3      | unproven|

So the model had a pretty tough time learning to classify things as "mixture" or "unproven". That would make a bit of sense because these categories are very much less clear-cut
than "true" or "false". I wondered if this was evidence for overfitting or underfitting, and so I decided to run an evaluation on the training set to further assess this. It
turned out that the training set accuracy was actually only 0.6308649530803754, which was lower than the validation set accuracy. I won't list the confusion matrix again, but it
had the same pattern of only predicting 0 and 2. My hypothesis for underfitting was a bit stronger now, so I decided to train the model for another epoch to see if this was 
indeed the case.
