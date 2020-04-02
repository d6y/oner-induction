// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The 1R (Holt, 1993) rule learning algorithm.
//!
//! # 1R is a baseline rule learning algorithm
//!
//! The algorithm generates a rule for each attribute in a dataset,
//! and then picks the "one rule" that has the best accuracy.
//!
//! Each rule (hypothesis) is a set of cases:
//! for every value of the attribute,
//! the prediction (the `then` part) is the most frequent class for examples with that attribute value.
//!
//! This is a baseline learner for use in comparison against more sophisticated algorithms.
//! A related idea is "0R" (zero rule), which is the most frequent class in the dataset.
//!
//! # Examples
//!
//! This crate uses [ndarray](https://docs.rs/ndarray/0.13.0/ndarray/) to represent attributes and classes.
//!
//! ```
//! use ndarray::prelude::*;
//! use oner_induction::{Rule, Case, Accuracy, discover};
//!
//! let examples = array![
//!    ["sunny", "summer"],
//!    ["sunny", "summer"],
//!    ["cloudy", "winter"],
//!    ["sunny", "winter"]
//! ];
//!
//! let classes = array![
//!     "hot",
//!     "hot",
//!     "cold",
//!     "cold"
//! ];
//!
//! // Discover the best rule, and the column it applies to:
//! let rule: Option<(usize, Rule<&str, &str>)> =
//!   discover(&examples.view(), &classes.view());
//!
//! // Expected accuracy is 100%
//! let accuracy = Accuracy(1.0);
//!
//! // The "rule" is a set of cases (conditions, or "IF...THENs"):
//! let cases = vec![
//!     Case { attribute_value: "summer", predicted_class: "hot" },
//!     Case { attribute_value: "winter", predicted_class: "cold" }
//! ];
//!
//! // Column 1 is the Season (winter or summer)
//! assert_eq!(rule, Some( (1, Rule { cases, accuracy }) ));
//! ```
//!
//! # References
//!
//! - Holte, R.C. (1993) Very Simple Classification Rules Perform Well on Most Commonly Used Datasets. _Machine Learning_ 11: 63. [https://doi.org/10.1023/A:1022631118932](https://doi.org/10.1023/A:1022631118932).
//! - Molnar, C, (2019) _Interpretable Machine Learning_. In particular: [Learn Rules from a Single Feature (OneR)](https://christophm.github.io/interpretable-ml-book/rules.html#learn-rules-from-a-single-feature-oner).
//!
//! # Terminology
//!
//! I'm following the terminology from Holte (1993):
//!
//! - Attribute (a.k.a. feature)
//! - Value (the value of an attribute or class)
//! - Class (classification, prediction)
//! - Example (instance)
//!
//! In generic parameters, `A` is for attribute and `C` is for class.
//!
//! # Limitations
//!
//! This crate assumes numeric data has already been converted to categorical data.
//!
//! See <https://docs.rs/oner_quantize> for an implementation of the 1R qualitzation algorithm.
//!

use std::hash::Hash;

/// A prediction based on an attribute value.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Case<A, C> {
    /// The attribute value this case matches against.
    pub attribute_value: A,
    /// The predicted class when the attribute value is matched.
    pub predicted_class: C,
}

/// The rule for an attribute, together with the training data accuracy.
#[derive(Debug, PartialEq)]
pub struct Rule<A, C> {
    /// The conditions and actions (IF...THENs) for an attribute.
    pub cases: Vec<Case<A, C>>,

    /// The accuracy of the rule set on the training data used to discover it.
    pub accuracy: Accuracy,
}

/// Fraction of correct predictions out of all rows in the training data.
#[derive(Debug, PartialEq, PartialOrd)]
pub struct Accuracy(pub f64);

mod induction;
pub use induction::discover;

mod evaluation;
pub use evaluation::{evaluate, interpret};
