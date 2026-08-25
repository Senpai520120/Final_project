"""URL configuration for the loan approval project."""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("loan_prediction.urls")),
]
