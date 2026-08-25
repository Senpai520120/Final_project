"""Tests for the loan prediction page and for the trained ML pipeline."""

import pandas as pd
from django.conf import settings
from django.test import SimpleTestCase
from django.urls import reverse

from loan_prediction.forms import LoanApplicationForm
from loan_prediction.views import get_model

VALID_APPLICATION = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": "5000",
    "CoapplicantIncome": "1500",
    "LoanAmount": "120",
    "Loan_Amount_Term": "360.0",
    "Credit_History": "1.0",
    "Property_Area": "Urban",
}


class HomePageTests(SimpleTestCase):
    def test_home_page_opens(self):
        response = self.client.get(reverse("loan_prediction:home"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "loan_prediction/home.html")

    def test_home_page_shows_every_form_field(self):
        response = self.client.get(reverse("loan_prediction:home"))
        for name in LoanApplicationForm().fields:
            self.assertContains(response, f'name="{name}"')


class PredictionTests(SimpleTestCase):
    def test_valid_application_returns_a_result(self):
        response = self.client.post(reverse("loan_prediction:home"), VALID_APPLICATION)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "loan_prediction/result.html")
        self.assertIn("approved", response.context)
        self.assertIsInstance(response.context["approved"], bool)

    def test_result_page_shows_a_verdict_and_a_probability(self):
        response = self.client.post(reverse("loan_prediction:home"), VALID_APPLICATION)
        verdict = "Loan approved" if response.context["approved"] else "Loan rejected"
        self.assertContains(response, verdict)
        self.assertContains(response, "Approval probability")
        self.assertGreaterEqual(response.context["probability"], 0)
        self.assertLessEqual(response.context["probability"], 100)

    def test_bad_credit_history_lowers_the_approval_probability(self):
        good = self.client.post(reverse("loan_prediction:home"), VALID_APPLICATION)
        bad = self.client.post(
            reverse("loan_prediction:home"),
            {**VALID_APPLICATION, "Credit_History": "0.0"},
        )
        self.assertLess(bad.context["probability"], good.context["probability"])


class FormValidationTests(SimpleTestCase):
    def test_empty_form_is_invalid(self):
        form = LoanApplicationForm(data={})
        self.assertFalse(form.is_valid())
        self.assertEqual(len(form.errors), len(form.fields))

    def test_negative_income_is_rejected(self):
        form = LoanApplicationForm(
            data={**VALID_APPLICATION, "ApplicantIncome": "-100"}
        )
        self.assertFalse(form.is_valid())
        self.assertIn("ApplicantIncome", form.errors)

    def test_unknown_choice_is_rejected(self):
        form = LoanApplicationForm(
            data={**VALID_APPLICATION, "Property_Area": "Atlantis"}
        )
        self.assertFalse(form.is_valid())
        self.assertIn("Property_Area", form.errors)

    def test_invalid_post_redisplays_the_form_with_errors(self):
        response = self.client.post(
            reverse("loan_prediction:home"),
            {**VALID_APPLICATION, "LoanAmount": "not a number"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "loan_prediction/home.html")
        self.assertContains(response, "The loan amount must be a number.")

    def test_dropdown_values_are_converted_to_floats(self):
        form = LoanApplicationForm(data=VALID_APPLICATION)
        self.assertTrue(form.is_valid())
        data = form.to_model_input()
        self.assertEqual(data["Credit_History"], 1.0)
        self.assertEqual(data["Loan_Amount_Term"], 360.0)


class PipelineTests(SimpleTestCase):
    def test_model_file_exists(self):
        self.assertTrue(
            settings.MODEL_FILE.exists(),
            "Run `python -m ml.train_model` before running the tests.",
        )

    def test_pipeline_contains_preprocessing_and_classifier(self):
        model = get_model()
        self.assertIn("preprocessor", model.named_steps)
        self.assertIn("classifier", model.named_steps)

    def test_pipeline_accepts_raw_form_data(self):
        form = LoanApplicationForm(data=VALID_APPLICATION)
        self.assertTrue(form.is_valid())
        row = pd.DataFrame([form.to_model_input()])

        model = get_model()
        self.assertIn(model.predict(row)[0], (0, 1))

        probabilities = model.predict_proba(row)[0]
        self.assertEqual(len(probabilities), 2)
        self.assertAlmostEqual(probabilities.sum(), 1.0, places=6)
