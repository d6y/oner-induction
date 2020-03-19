// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use super::{Accuracy, Case};
use ndarray::{ArrayView, Ix1, Zip};

/// Apply a set of cases to an attribute value to get a prediction.
///
/// # Examples
///
/// ```
/// use oner_induction::{Case, interpret};
///
/// let cases = vec![
///     Case { attribute_value: "summer", predicted_class: "hot" },
///     Case { attribute_value: "winter", predicted_class: "cold" },
/// ];
///
/// assert_eq!( Some(&"hot"), interpret(&cases, &"summer"));
/// assert_eq!( None, interpret(&cases, &"spring"));
/// ```
pub fn interpret<'c, A: PartialEq, C>(
    cases: &'c [Case<A, C>],
    attribute_value: &A,
) -> Option<&'c C> {
    cases
        .iter()
        .find(|case| &case.attribute_value == attribute_value)
        .map(|case| &case.predicted_class)
}

/// Evaluate cases (a.k.a., a rule) against a data set, to get a performance accuracy.
///
/// Accuracy is defined as the number of correct predictions over the number of rows.
pub fn evaluate<A: PartialEq, C: PartialEq>(
    cases: &[Case<A, C>],
    attribute_values: &ArrayView<A, Ix1>,
    classes: &ArrayView<C, Ix1>,
) -> Accuracy {
    let mut right_wrong: Vec<Option<bool>> = Vec::new();

    Zip::from(attribute_values).and(classes).apply(|attribute_value, class| {
        match interpret(cases, attribute_value) {
            None => right_wrong.push(None),
            Some(predicted) => right_wrong.push(Some(predicted == class)),
        }
    });

    let num_examples = classes.len();

    if num_examples == 0 {
        Accuracy(0.0)
    } else {
        let num_correct = right_wrong.into_iter().filter(|&o| o == Some(true)).count();
        Accuracy(num_correct as f64 / num_examples as f64)
    }
}
