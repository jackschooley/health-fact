# health-fact

## Overview

The goal of this project was to use one of Hugging Face's pretrained transformer models to classify their `health_fact` dataset. This dataset has a few features which correspond
to nontrivial amounts of natural language, among them the `main_text`, `claim`, and `explanation`. I took an approach kind of similar to question answering with regard to the 
text I used, which was to essentially just take the claim and concatenate it with the explanation. I thought this would be the most prudent initial path to take because it could
capture the most meaning which relates to the label while being succint. The `main_text` feature especially was extremely long, and so I didn't think it was a good idea to use 
it because I would probably be forced to truncate the vast majority of it anyway.

I used the base DistilBERT as my transformer of choice. For me it was the natural choice because I have the most experience using it, and it does have the nice property of
being a relatively more lightweight transformer. After running the model inputs through DistilBERT, I retrieved the last hidden state for the leading `[CLS]` token and fed it
through to a linear layer to get the logits for all 4 potential labels. I thought this would be a relatively simple yet effective approach.

## Preprocessing

Luckily, this dataset was pretty easy to preprocess for the most part. There were only a handful of "unclean" observations that were actually labeled `-1`, which doesn't 
correspond to any category. I took a look at a couple of these observations and they were not really coherent at all in the context of the rest of the dataset, so I decided to 
just drop them during both training and evaluation. I set up a pipeline that would fetch the relevant dataset split, remove the bad observations as described, and tokenize the 
concatenated claim and explanation (padding and truncating if necessary). This pipeline would yield the necessary input IDs, attention masks, and labels for both training and
validation.

## Training

Training was relatively simple and standard as well. I used stochastic gradient descent as my optimization algorithm and cross-entropy loss as my loss function. What might seem
a little weird though was that I only used a batch size of 2. This is really small, but the reason I did this was so that I could run the training loop locally on my computer
without running out of memory on my GPU. Of course, I could have just used Google Colab or something to get more GPU memory, but I find it more fun to host all the computation
locally for small projects.

## Results

First I trained the model for one epoch. This led to a validation set accuracy of 0.652. What was most interesting about these results, though, was that the model only
predicted 0 or 2 and never 1 or 3. For reference, here are the corresponding labels for each class number.

| Number | Meaning |
|--------|---------|
| 0      | false   |
| 1      | mixture |
| 2      | true    |
| 3      | unproven|

So the model had a pretty tough time learning to classify things as `mixture` or `unproven`. That would make a bit of sense because these categories are very much less clear-cut
than `true` or `false`. I wondered if this was evidence for overfitting or underfitting, and so I decided to run an evaluation on the training set to further assess this. It
turned out that the training set accuracy was actually only 0.631, which was lower than the validation set accuracy. The model predicted only 0 and 2 on the training set as 
well, which adds to the evidence in favor of underfitting, so I decided to train the model for another epoch to test this hypothesis.

|   | 0    | 1  | 2    | 3 |
|---|------|----|------|---|
| 0 | 299  | 7  | 74   | 0 |
| 1 | 115  | 11 | 38   | 0 |
| 2 | 103  | 11 | 515  | 0 |
| 3 | 31   | 1  | 9    | 0 |

Here is the confusion matrix for the validation set after two epochs of training. Training the model for a second epoch did lead to an improvement in accuracy to 0.680. Training 
accuracy also increased to 0.676, and the model is starting to predict 1s for that split as well. This is evidence that the model is continuing to learn with more training data 
and has not yet run into overfitting issues. Finally, after this improvement I used the test set for a final evaluation.

|   | 0    | 1  | 2    | 3 |
|---|------|----|------|---|
| 0 | 309  | 5  | 74   | 0 |
| 1 | 128  | 12 | 61   | 0 |
| 2 | 120  | 14 | 465  | 0 |
| 3 | 32   | 0  | 13   | 0 |

Here is the confusion matrix for the test set. Unfortunately, the accuracy on the test set is 0.637, which is lower than on the validation set, but it's in the same ballpark 
overall.

## Conclusion

Overall, this straightforward modeling approach and architecture was able to get an accuracy of approximately 2/3 in a 4-class classification problem using only the claim and 
explanation natural language features. Greater performance could probably be attained using this same architecture only by training for even more epochs.

Other improvements could be to incorporate more of the data or improve on the model architecture. The first place I would go to yield more data would probably be the subjects, 
as these would be short and sweet language features that might still yield some interesting insights. Perhaps certain subjects are more prone to belonging to one class. As far 
as architecture improvements, more linear layers would be an obvious thing to try, as the classification head is the part of the model that is not pretrained and thus could use 
more finetuning and complexity. Finally, a more sophisticated pooling layer could be taken for each sequence of the DistilBERT last hidden states, instead of just taking the 
`[CLS]` token.
