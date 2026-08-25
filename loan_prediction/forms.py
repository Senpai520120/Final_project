"""Input form for the loan approval prediction.

Every field matches a column of data/loan_data.csv and every choice list
contains exactly the values that occur in that dataset, so the data the user
types can be fed to the trained pipeline unchanged.
"""

from django import forms

GENDER_CHOICES = [("Male", "Male"), ("Female", "Female")]
YES_NO_CHOICES = [("Yes", "Yes"), ("No", "No")]
DEPENDENTS_CHOICES = [("0", "0"), ("1", "1"), ("2", "2"), ("3+", "3 or more")]
EDUCATION_CHOICES = [("Graduate", "Graduate"), ("Not Graduate", "Not graduate")]
CREDIT_HISTORY_CHOICES = [
    ("1.0", "Good — no problems with previous loans"),
    ("0.0", "Bad — problems with previous loans"),
]
PROPERTY_AREA_CHOICES = [
    ("Urban", "Urban"),
    ("Semiurban", "Semiurban"),
    ("Rural", "Rural"),
]

# Loan_Amount_Term is expressed in months and only takes a handful of values
# in the dataset; the most common one by far is 360 (30 years).
LOAN_TERM_CHOICES = [
    ("360.0", "360 months (30 years)"),
    ("480.0", "480 months (40 years)"),
    ("300.0", "300 months (25 years)"),
    ("240.0", "240 months (20 years)"),
    ("180.0", "180 months (15 years)"),
    ("120.0", "120 months (10 years)"),
    ("84.0", "84 months (7 years)"),
    ("60.0", "60 months (5 years)"),
    ("36.0", "36 months (3 years)"),
    ("12.0", "12 months (1 year)"),
]


class LoanApplicationForm(forms.Form):
    """Collects one loan application and returns it as pipeline-ready data."""

    Gender = forms.ChoiceField(
        label="Gender",
        choices=GENDER_CHOICES,
        error_messages={"required": "Please select a gender."},
    )
    Married = forms.ChoiceField(
        label="Married",
        choices=YES_NO_CHOICES,
        error_messages={"required": "Please state whether the applicant is married."},
    )
    Dependents = forms.ChoiceField(
        label="Number of dependents",
        choices=DEPENDENTS_CHOICES,
        error_messages={"required": "Please select the number of dependents."},
    )
    Education = forms.ChoiceField(
        label="Education",
        choices=EDUCATION_CHOICES,
        error_messages={"required": "Please select an education level."},
    )
    Self_Employed = forms.ChoiceField(
        label="Self-employed",
        choices=YES_NO_CHOICES,
        error_messages={"required": "Please state whether the applicant is self-employed."},
    )
    ApplicantIncome = forms.IntegerField(
        label="Applicant monthly income",
        min_value=0,
        max_value=1_000_000,
        help_text="Monthly income of the main applicant.",
        error_messages={
            "required": "Please enter the applicant income.",
            "invalid": "The applicant income must be a whole number.",
            "min_value": "The applicant income cannot be negative.",
        },
    )
    CoapplicantIncome = forms.FloatField(
        label="Co-applicant monthly income",
        min_value=0,
        max_value=1_000_000,
        initial=0,
        help_text="Enter 0 if there is no co-applicant.",
        error_messages={
            "required": "Please enter the co-applicant income (0 if there is none).",
            "invalid": "The co-applicant income must be a number.",
            "min_value": "The co-applicant income cannot be negative.",
        },
    )
    LoanAmount = forms.FloatField(
        label="Loan amount (in thousands)",
        min_value=1,
        max_value=10_000,
        help_text="Requested amount, in thousands — the same unit as the dataset.",
        error_messages={
            "required": "Please enter the loan amount.",
            "invalid": "The loan amount must be a number.",
            "min_value": "The loan amount must be greater than zero.",
        },
    )
    Loan_Amount_Term = forms.ChoiceField(
        label="Loan term",
        choices=LOAN_TERM_CHOICES,
        initial="360.0",
        error_messages={"required": "Please select a loan term."},
    )
    Credit_History = forms.ChoiceField(
        label="Credit history",
        choices=CREDIT_HISTORY_CHOICES,
        initial="1.0",
        error_messages={"required": "Please select the credit history."},
    )
    Property_Area = forms.ChoiceField(
        label="Property area",
        choices=PROPERTY_AREA_CHOICES,
        error_messages={"required": "Please select a property area."},
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # One shared CSS class keeps the template markup simple.
        for field in self.fields.values():
            css_class = "select" if isinstance(field, forms.ChoiceField) else "input"
            field.widget.attrs["class"] = css_class

    def to_model_input(self):
        """Return the cleaned data with the exact types the pipeline expects.

        The two numeric columns that are presented as dropdowns
        (Loan_Amount_Term and Credit_History) arrive as strings and have to be
        converted back to floats, because that is how they were stored during
        training.
        """
        data = dict(self.cleaned_data)
        data["Loan_Amount_Term"] = float(data["Loan_Amount_Term"])
        data["Credit_History"] = float(data["Credit_History"])
        return data

    def summary(self):
        """Return the submitted application as readable (label, value) pairs.

        The result page shows what the user filled in, so it has to use the
        field labels and the choice labels rather than the raw dataset column
        names and their encoded values.
        """
        rows = []
        for name, field in self.fields.items():
            value = self.cleaned_data[name]
            if isinstance(field, forms.ChoiceField):
                value = dict(field.choices).get(value, value)
            rows.append((field.label, value))
        return rows
