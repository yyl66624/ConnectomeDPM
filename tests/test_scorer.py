from connectomedpm.supervision.scorer import parse_number, score_answer


def test_numeric_equivalence():
    assert score_answer("1010", "1010", "numerical_arithmetic")[0]
    assert score_answer("Answer: 1010", "1010", "numerical_arithmetic")[0]
    assert score_answer("1,010", "1010", "numerical_arithmetic")[0]
    assert score_answer("1010.0", "1010", "numerical_arithmetic")[0]
    assert not score_answer("1011", "1010", "numerical_arithmetic")[0]


def test_verbose_output_takes_first_line():
    assert score_answer("1010\nbecause 500+510", "1010", "numerical_arithmetic")[0]


def test_text_answers():
    assert score_answer("even", "EVEN", "format_following")[0]
    assert score_answer("The answer is EVEN.", "EVEN", "format_following")[0]
    assert not score_answer("ODD", "EVEN", "format_following")[0]


def test_empty_prediction_is_wrong():
    assert not score_answer("", "42", "numerical_arithmetic")[0]
    assert not score_answer("   ", "42", "numerical_arithmetic")[0]


def test_parse_number_tolerates_separators():
    assert parse_number("1,234") == 1234.0
    assert parse_number("x = -7") == -7.0
    assert parse_number("no digits here") is None

