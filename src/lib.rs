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

#[derive(Debug, PartialEq, PartialOrd)]
pub struct Accuracy(pub f64);

mod induction;
pub use induction::discover;

mod evaluation;
pub use evaluation::{evaluate, interpret};