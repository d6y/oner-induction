//! The 1R (Holt, 1993) rule learning algorithm.
//!
//! # 1R is a baseline rule learning algorithm
//!
//! The algorithm generates a rule for each attribute in a dataset,
//! and then picks the "one rule" that has the best accuracy.
//!
//! Each rule (hypothesis) is: for every value of the attribute,
//! the prediction (the `then` part) is the most frequent class (that has the attribute value).
//!
//! A related idea is "0R" (zero rule), which is the most frequent class in the dataset.
//!
//! # Examples
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
//! - Holte, R.C. _Machine Learning_ (1993) 11: 63. [https://doi.org/10.1023/A:1022631118932](https://doi.org/10.1023/A:1022631118932).
//! - Molnar, C, _Interpretable Machine Learning_ (2019). In particular: [Learn Rules from a Single Feature (OneR)](https://christophm.github.io/interpretable-ml-book/rules.html#learn-rules-from-a-single-feature-oner).
//!
//! ## Terminology
//!
//! I'm following the terminology from Holte (1993):
//!
//! - Attribute (a.k.a. feature)
//! - Value (the value of an attribute or class)
//! - Class (classification, prediction)
//! - Example (instance)
//!
//! In generic parameters, `A` is for attribute and `C` is for class.

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
