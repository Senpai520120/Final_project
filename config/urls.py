"""URL configuration for the loan approval project."""

from django.urls import include, path

urlpatterns = [
    path("", include("loan_prediction.urls")),
]
