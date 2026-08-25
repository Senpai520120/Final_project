"""Views for the loan approval prediction page."""

import functools
import logging

import joblib
import pandas as pd
from django.conf import settings
from django.shortcuts import render

from loan_prediction.forms import LoanApplicationForm

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def get_model():
    """Load the trained pipeline once and reuse it for every request."""
    return joblib.load(settings.MODEL_FILE)


def predict_loan(form):
    """Run one validated application through the pipeline.

    The saved pipeline already contains the imputation, scaling and encoding
    steps, so the raw form values can be handed over as they are.
    """
    row = pd.DataFrame([form.to_model_input()])
    model = get_model()
    approved = bool(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1]) * 100
    return approved, probability


def home(request):
    """Show the application form and, on POST, the prediction result."""
    if request.method != "POST":
        return render(request, "loan_prediction/home.html",
                      {"form": LoanApplicationForm()})

    form = LoanApplicationForm(request.POST)
    if not form.is_valid():
        return render(request, "loan_prediction/home.html", {"form": form})

    try:
        approved, probability = predict_loan(form)
    except Exception:
        # Never expose a traceback to the user; log it for the developer.
        logger.exception("Loan prediction failed")
        return render(request, "loan_prediction/home.html", {
            "form": form,
            "error": "The prediction service is currently unavailable. "
                     "Please try again later.",
        })

    return render(request, "loan_prediction/result.html", {
        "approved": approved,
        "probability": round(probability, 1),
        "application": form.cleaned_data,
    })
