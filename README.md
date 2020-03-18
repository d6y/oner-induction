# A 1R implementation in Rust

Re-implementing the 1R algorithm described in Holte (1993).

## What is this?

1R is a baseline rule learning algorithm.

The algorithm generates a rule for each attribute, and then picks the "one rule" that has the best accuracy.

Each rule (hypothesis) is: for every value of the attribute, the prediction (the `then` part) is the most frequent class (that has the attribute value).

For example, given a data set of drinking habits with attributes such as age, time of day, mood (attributes), 1R might produces a rule of the form:

```
if time="morning" then drink="coffee"
if time="afternoon" then drink="tea"
if time="evening" then drink="water"
```

The rule might only have, say, 60% accuracy. 
That's a baseline to compare to other algorithms.

A related idea is "0R" (zero rule), which is the most frequent class in the dataset.
That is, if our drinking habits data has 100 rows, and 51 of them were for "tea", 
then 0R would be: predict "tea" (and would have an accuracy of 51/100).

See:

- Holte, R.C. _Machine Learning_ (1993) 11: 63. [https://doi.org/10.1023/A:1022631118932](https://doi.org/10.1023/A:1022631118932).

- Molnar, C, _Interpretable Machine Learning_ (2019). In particular: [Learn Rules from a Single Feature (OneR)](https://christophm.github.io/interpretable-ml-book/rules.html#learn-rules-from-a-single-feature-oner).


## Crates

## Terminology

I'm following the terminology from Holte (1993):

- Attribute (a.k.a. feature)
- Value (the value of an attribute or class)
- Class (a.k.a. classification, prediction)
- Example (a.k.a. instance)

## Example data sets

I have taken data sets and converted to CSV where necessary, including adding header rows.

The `data` folder contains the data from various sources. Unless otherwise specified, it'll be the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/citation_policy.html).

- `bc`, a [breast cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer) dataset.
   In the CSV version I have moved the class from the first column to the last column because that's what this code expects. I did this with: `awk -F, '{print $2,$3,$4,$5,$6,$7,$8,$9,$10,$1}' OFS=, < breast-cancer.data  > bc.csv`

- `ch`, the [Chess (King-Rook vs. King-Pawn)](https://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King-Pawn%29) dataset.

- `cc`, the [Cervical cancer (risk Factors)](https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29) dataset. I've removed the Hinselmann, Schiller, and Cytology targets, leaving just the Biopsy target.

- `fake-house`, the dataset used to introduce 1R in [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/rules.html#learn-rules-from-a-single-feature-oner) (published under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)). To run the example use the `-w` flag to use the whole dataset for rule discovery.

## Documentation

To open the internal documentation:

```
$ cargo doc --no-deps --open --document-private-items
```
